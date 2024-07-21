# `.\pytorch\torch\csrc\api\include\torch\optim\rmsprop.h`

```
#pragma once

#include <torch/nn/module.h>  // 引入 Torch 深度学习框架的模块定义
#include <torch/optim/optimizer.h>  // 引入 Torch 优化器的定义
#include <torch/optim/serialize.h>  // 引入 Torch 优化器序列化相关的定义
#include <torch/serialize/archive.h>  // 引入 Torch 序列化存档相关的定义
#include <torch/types.h>  // 引入 Torch 的数据类型定义

#include <functional>  // 引入 C++ 标准库的函数对象
#include <memory>  // 引入 C++ 标准库的内存管理
#include <string>  // 引入 C++ 标准库的字符串
#include <vector>  // 引入 C++ 标准库的向量容器

namespace torch {
namespace serialize {
class OutputArchive;  // Torch 序列化输出存档类的前置声明
class InputArchive;  // Torch 序列化输入存档类的前置声明
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

struct TORCH_API RMSpropOptions  // RMSprop 优化器的选项定义
    : public OptimizerCloneableOptions<RMSpropOptions> {
  RMSpropOptions(double lr = 1e-2);  // 构造函数，默认学习率为 0.01
  TORCH_ARG(double, lr) = 1e-2;  // 学习率参数，默认值为 0.01
  TORCH_ARG(double, alpha) = 0.99;  // RMSprop 算法中的衰减率参数，默认值为 0.99
  TORCH_ARG(double, eps) = 1e-8;  // 数值稳定性参数，默认值为 1e-8
  TORCH_ARG(double, weight_decay) = 0;  // 权重衰减参数，默认值为 0
  TORCH_ARG(double, momentum) = 0;  // 动量参数，默认值为 0
  TORCH_ARG(bool, centered) = false;  // 是否启用中心化参数，默认为 false

 public:
  void serialize(torch::serialize::InputArchive& archive) override;  // 序列化输入存档函数的重写声明
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 序列化输出存档函数的重写声明
  TORCH_API friend bool operator==(  // 等号操作符的友元声明
      const RMSpropOptions& lhs,
      const RMSpropOptions& rhs);
  double get_lr() const override;  // 获取学习率的函数声明
  void set_lr(const double lr) override;  // 设置学习率的函数声明
};

struct TORCH_API RMSpropParamState  // RMSprop 优化器的参数状态定义
    : public OptimizerCloneableParamState<RMSpropParamState> {
  TORCH_ARG(int64_t, step) = 0;  // 步数参数，默认值为 0
  TORCH_ARG(torch::Tensor, square_avg);  // 平方均值参数
  TORCH_ARG(torch::Tensor, momentum_buffer) = {};  // 动量缓冲区参数，默认为空
  TORCH_ARG(torch::Tensor, grad_avg) = {};  // 梯度平均值参数，默认为空

 public:
  void serialize(torch::serialize::InputArchive& archive) override;  // 序列化输入存档函数的重写声明
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 序列化输出存档函数的重写声明
  TORCH_API friend bool operator==(  // 等号操作符的友元声明
      const RMSpropParamState& lhs,
      const RMSpropParamState& rhs);
};

class TORCH_API RMSprop : public Optimizer {  // RMSprop 优化器的实现类，继承自 Optimizer
 public:
  explicit RMSprop(  // 显式构造函数声明，接受参数组和默认选项
      std::vector<OptimizerParamGroup> param_groups,
      RMSpropOptions defaults = {})
      : Optimizer(
            std::move(param_groups),
            std::make_unique<RMSpropOptions>(defaults)) {  // 调用基类 Optimizer 的构造函数
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());  // 检查学习率参数是否合法
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());  // 检查数值稳定性参数是否合法
    TORCH_CHECK(
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());  // 检查动量参数是否合法
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());  // 检查权重衰减参数是否合法
    TORCH_CHECK(
        defaults.alpha() >= 0, "Invalid alpha value: ", defaults.alpha());  // 检查衰减率参数是否合法
  }

  explicit RMSprop(std::vector<Tensor> params, RMSpropOptions defaults = {})  // 显式构造函数声明，接受参数列表和默认选项
      : RMSprop({OptimizerParamGroup(std::move(params))}, defaults) {}  // 调用另一个构造函数进行初始化

  torch::Tensor step(LossClosure closure = nullptr) override;  // 执行一步优化更新的函数声明
  void save(serialize::OutputArchive& archive) const override;  // 保存模型状态到输出存档的函数声明
  void load(serialize::InputArchive& archive) override;  // 从输入存档加载模型状态的函数声明

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {  // 静态模板序列化函数的定义
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(RMSprop);  // 使用模板参数序列化
  }
};
} // namespace optim
} // namespace torch
```