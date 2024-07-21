# `.\pytorch\torch\csrc\api\include\torch\optim\adam.h`

```
#pragma once

#include <torch/nn/module.h>  // 引入 Torch 的神经网络模块
#include <torch/optim/optimizer.h>  // 引入 Torch 的优化器接口
#include <torch/optim/serialize.h>  // 引入 Torch 的序列化接口

#include <utility>  // 引入 std::utility
#include <vector>  // 引入 std::vector

namespace torch {
namespace serialize {
class OutputArchive;  // Torch 序列化输出存档类声明
class InputArchive;   // Torch 序列化输入存档类声明
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

struct TORCH_API AdamOptions : public OptimizerCloneableOptions<AdamOptions> {
  AdamOptions(double lr = 1e-3);  // Adam 优化器选项类的构造函数声明，默认学习率为 0.001
  TORCH_ARG(double, lr) = 1e-3;   // 学习率参数，默认值为 0.001
  typedef std::tuple<double, double> betas_t;  // 定义双 beta 参数的元组类型
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);  // 默认的 beta 参数值为 (0.9, 0.999)
  TORCH_ARG(double, eps) = 1e-8;  // 默认的 epsilon 参数值为 1e-8
  TORCH_ARG(double, weight_decay) = 0;  // 默认的权重衰减参数值为 0
  TORCH_ARG(bool, amsgrad) = false;  // 默认不启用 AMSGrad 变种

 public:
  void serialize(torch::serialize::InputArchive& archive) override;   // 序列化输入存档方法声明
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 序列化输出存档方法声明
  TORCH_API friend bool operator==(   // 友元函数声明，用于比较两个 AdamOptions 对象是否相等
      const AdamOptions& lhs,
      const AdamOptions& rhs);
  double get_lr() const override;  // 获取当前学习率的方法声明
  void set_lr(const double lr) override;  // 设置学习率的方法声明
};

struct TORCH_API AdamParamState  // Adam 优化器参数状态类声明
    : public OptimizerCloneableParamState<AdamParamState> {
  TORCH_ARG(int64_t, step) = 0;   // 优化器的步数参数，默认为 0
  TORCH_ARG(torch::Tensor, exp_avg);  // 指数移动平均值的张量参数
  TORCH_ARG(torch::Tensor, exp_avg_sq);  // 指数移动平方平均值的张量参数
  TORCH_ARG(torch::Tensor, max_exp_avg_sq) = {};  // 最大指数移动平方平均值的张量参数，默认为空

 public:
  void serialize(torch::serialize::InputArchive& archive) override;   // 序列化输入存档方法声明
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 序列化输出存档方法声明
  TORCH_API friend bool operator==(   // 友元函数声明，用于比较两个 AdamParamState 对象是否相等
      const AdamParamState& lhs,
      const AdamParamState& rhs);
};

class TORCH_API Adam : public Optimizer {  // Adam 优化器类声明，继承自 Optimizer 类
 public:
  explicit Adam(   // 显式构造函数声明，接受参数组列表和默认选项
      std::vector<OptimizerParamGroup> param_groups,
      AdamOptions defaults = {})
      : Optimizer(   // 调用基类 Optimizer 的构造函数
            std::move(param_groups),
            std::make_unique<AdamOptions>(defaults)) {
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());  // 检查学习率是否合法
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());  // 检查 epsilon 是否合法
    auto betas = defaults.betas();  // 获取默认的 beta 参数
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));  // 检查第一个 beta 参数是否合法
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));  // 检查第二个 beta 参数是否合法
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());  // 检查权重衰减参数是否合法
  }
  explicit Adam(std::vector<Tensor> params, AdamOptions defaults = {})   // 显式构造函数声明，接受张量列表和默认选项
      : Adam({OptimizerParamGroup(std::move(params))}, defaults) {}  // 委托构造函数

  torch::Tensor step(LossClosure closure = nullptr) override;  // 执行优化步骤的方法声明
  void save(serialize::OutputArchive& archive) const override;  // 保存模型状态的方法声明
  void load(serialize::InputArchive& archive) override;  // 加载模型状态的方法声明

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {   // 静态模板序列化方法声明
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adam);  // 调用 Torch 提供的优化器序列化模板宏
  }
};
} // namespace optim
} // namespace torch
```