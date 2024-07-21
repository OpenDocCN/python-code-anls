# `.\pytorch\torch\csrc\api\src\optim\schedulers\reduce_on_plateau_scheduler.cpp`

```
// 引入 Torch 库中的优化器模块和调度器模块的头文件
#include <torch/optim/schedulers/reduce_on_plateau_scheduler.h>

// 引入 C++ 标准库中的输入输出操作支持
#include <iomanip>

// Torch 命名空间
namespace torch {
namespace optim {

// ReduceLROnPlateauScheduler 类的构造函数定义，初始化成员变量和参数
ReduceLROnPlateauScheduler::ReduceLROnPlateauScheduler(
    Optimizer& optimizer,  // 传入的优化器引用
    SchedulerMode mode,    // 调度器模式
    float factor,          // 学习率调整因子
    int patience,          // 容忍的无效 epoch 数量
    double threshold,      // 阈值
    ThresholdMode threshold_mode,  // 阈值模式
    int cooldown,          // 冷却期
    const std::vector<float>& min_lr,  // 最小学习率向量
    double eps,            // 用于比较浮点数的小值
    bool verbose)          // 是否显示详细信息
    : optimizer(optimizer) {  // 初始化列表中初始化优化器成员

  // 如果最小学习率向量为空，则创建一个与参数组数目相同的默认向量
  if (min_lr.empty()) {
    this->min_lrs = std::vector<float>(optimizer.param_groups().size());
  } else {
    // 检查最小学习率向量的长度与优化器中参数组的数量是否一致
    TORCH_CHECK(
        min_lr.size() == optimizer.param_groups().size(),
        "Number of learning rates not equal to the number of param groups\n",
        "Number of learning rates given: ",
        min_lr.size(),
        "\nNumber of param groups: ",
        optimizer.param_groups().size());
    this->min_lrs = min_lr;
  }

  // 检查学习率调整因子是否小于 1.0
  TORCH_CHECK(factor < 1.0, "Factor should be < 1.0.");

  // 初始化成员变量
  this->factor = factor;
  this->patience = patience;
  this->cooldown = cooldown;
  this->eps = eps;
  this->verbose = verbose;

  // 初始化比较函数
  init_is_better(mode, threshold, threshold_mode);

  // 重置内部状态
  reset();
}

// 调度器的一步更新函数，根据指标值调整学习率
void ReduceLROnPlateauScheduler::step(float metrics) {
  last_epoch++;  // 增加当前 epoch 计数

  // 如果当前指标值比历史最佳值好，则更新最佳指标和无效 epoch 计数
  if (is_better(metrics)) {
    best = metrics;
    num_bad_epochs = 0;
  } else {
    num_bad_epochs++;  // 否则增加无效 epoch 计数
  }

  // 如果在冷却期内，则递减冷却计数器并重置无效 epoch 计数
  if (in_cooldown()) {
    cooldown_counter--;
    num_bad_epochs = 0;
  }

  // 如果连续无效 epoch 达到容忍上限，则降低学习率
  if (num_bad_epochs > patience) {
    reduce_lr(last_epoch);  // 调用学习率降低函数
    cooldown_counter = cooldown;  // 进入冷却期
    num_bad_epochs = 0;  // 重置无效 epoch 计数
  }
}

// 实际降低学习率的函数，根据当前 epoch 和条件对每个参数组进行调整
void ReduceLROnPlateauScheduler::reduce_lr(int epoch) {
  for (std::size_t i = 0; i < optimizer.param_groups().size(); i++) {
    auto old_lr = optimizer.param_groups()[i].options().get_lr();  // 获取旧的学习率
    auto new_lr = std::fmax(old_lr * factor, min_lrs[i]);  // 计算新的学习率
    if (old_lr - new_lr > eps) {  // 如果变化大于设定的 epsilon，则更新学习率
      optimizer.param_groups()[i].options().set_lr(new_lr);
      if (verbose) {
        // 如果设置了详细输出，则显示学习率的变化信息
        std::cout << std::setprecision(4) << "Epoch " << epoch
                  << ": reducing learning rate of group " << i << " to "
                  << new_lr << std::endl;
      }
    }
  }
}

// 重置调度器的内部状态
void ReduceLROnPlateauScheduler::reset() {
  this->cooldown_counter = 0;  // 冷却计数器清零
  this->num_bad_epochs = 0;   // 无效 epoch 计数清零
  this->last_epoch = 0;       // 最后一个 epoch 清零
  this->best = mode_worse;    // 最佳指标初始化
}

// 判断当前是否在冷却期
bool ReduceLROnPlateauScheduler::in_cooldown() {
  return cooldown_counter > 0;  // 冷却计数器大于 0 则处于冷却期
}

// 根据当前模式和阈值判断当前指标值是否优于历史最佳值
bool ReduceLROnPlateauScheduler::is_better(float a) {
  if (mode == min && threshold_mode == rel) {
    auto rel_epsilon = 1.0 - threshold;
    return a < best * rel_epsilon;  // 相对模式下判断是否优于阈值比例
  } else if (mode == min && threshold_mode == abs) {
    return a < best - threshold;  // 绝对模式下判断是否优于固定阈值
  } else if (mode == max && threshold_mode == rel) {
    auto rel_epsilon = 1.0 + threshold;
    return a > best * rel_epsilon;  // 相对模式下判断是否优于阈值比例
  } else {
    return a > best * threshold;  // 默认情况下判断是否优于固定阈值
  }
}

// 初始化比较函数的辅助函数，根据模式和阈值初始化比较逻辑
void ReduceLROnPlateauScheduler::init_is_better(
    SchedulerMode mode,  // 模式（最小或最大）
    double threshold,    // 阈值
    ThresholdMode threshold_mode) {  // 阈值模式
  // 根据模式和阈值模式初始化比较逻辑
  // 实现略去，部分代码未提供
}

// Torch 命名空间的结束
}  // namespace optim
}  // namespace torch
    ThresholdMode threshold_mode) {
  // 检查传入的 mode 是否为 min
  if (mode == min) {
    // 如果 mode 是 min，则将 mode_worse 设置为 float 类型的最大值
    mode_worse = std::numeric_limits<float>::max();
  } else {
    // 如果 mode 不是 min，则将 mode_worse 设置为 float 类型的最小值
    mode_worse = std::numeric_limits<float>::min();
  }

  // 将当前函数的 mode 参数赋值给对象的 mode 成员变量
  this->mode = mode;
  // 将当前函数的 threshold_mode 参数赋值给对象的 threshold_mode 成员变量
  this->threshold_mode = threshold_mode;
  // 将当前函数的 threshold 参数赋值给对象的 threshold 成员变量
  this->threshold = threshold;
}
}
} // namespace optim
} // namespace torch
```