# `.\pytorch\torch\csrc\api\include\torch\optim\adamw.h`

```py
#pragma once

#include <torch/nn/module.h>  // 引入 PyTorch 的神经网络模块
#include <torch/optim/optimizer.h>  // 引入 PyTorch 的优化器基类
#include <torch/optim/serialize.h>  // 引入 PyTorch 的序列化工具

#include <utility>  // 引入 C++ 实用工具库
#include <vector>   // 引入 C++ 向量容器

namespace torch {
namespace serialize {
class OutputArchive;  // 声明输出存档类，用于序列化
class InputArchive;   // 声明输入存档类，用于反序列化
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

struct TORCH_API AdamWOptions : public OptimizerCloneableOptions<AdamWOptions> {
  AdamWOptions(double lr = 1e-3);  // 构造函数，设定学习率默认值
  TORCH_ARG(double, lr) = 1e-3;  // 学习率参数，默认为 0.001
  typedef std::tuple<double, double> betas_t;  // 定义元组类型，用于存储 beta 参数
  TORCH_ARG(betas_t, betas) = std::make_tuple(0.9, 0.999);  // 默认的 beta 参数
  TORCH_ARG(double, eps) = 1e-8;  // epsilon 参数，默认为 1e-8
  TORCH_ARG(double, weight_decay) = 1e-2;  // 权重衰减参数，默认为 0.01
  TORCH_ARG(bool, amsgrad) = false;  // 是否使用 AMSGrad，默认为 false

 public:
  void serialize(torch::serialize::InputArchive& archive) override;  // 序列化函数声明
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 序列化函数声明
  TORCH_API friend bool operator==(  // 相等运算符重载声明
      const AdamWOptions& lhs,
      const AdamWOptions& rhs);
  double get_lr() const override;  // 获取学习率函数声明
  void set_lr(const double lr) override;  // 设置学习率函数声明
};

struct TORCH_API AdamWParamState
    : public OptimizerCloneableParamState<AdamWParamState> {
  TORCH_ARG(int64_t, step) = 0;  // 步数参数，默认为 0
  TORCH_ARG(torch::Tensor, exp_avg);  // 指数平均值参数
  TORCH_ARG(torch::Tensor, exp_avg_sq);  // 指数平方平均值参数
  TORCH_ARG(torch::Tensor, max_exp_avg_sq) = {};  // 最大指数平方平均值参数

 public:
  void serialize(torch::serialize::InputArchive& archive) override;  // 序列化函数声明
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 序列化函数声明
  TORCH_API friend bool operator==(  // 相等运算符重载声明
      const AdamWParamState& lhs,
      const AdamWParamState& rhs);
};

class TORCH_API AdamW : public Optimizer {
 public:
  explicit AdamW(
      std::vector<OptimizerParamGroup> param_groups,  // 参数组向量，用于初始化优化器
      AdamWOptions defaults = {})  // 默认选项，用于设定参数
      : Optimizer(
            std::move(param_groups),  // 转移参数组向量
            std::make_unique<AdamWOptions>(defaults)) {  // 创建唯一的选项对象
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());  // 检查学习率是否合法
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());  // 检查 epsilon 值是否合法
    auto betas = defaults.betas();  // 获取 beta 参数
    TORCH_CHECK(
        0 <= std::get<0>(betas) && std::get<0>(betas) < 1.0,  // 检查 beta 参数的合法性
        "Invalid beta parameter at index 0: ",
        std::get<0>(betas));
    TORCH_CHECK(
        0 <= std::get<1>(betas) && std::get<1>(betas) < 1.0,  // 检查 beta 参数的合法性
        "Invalid beta parameter at index 1: ",
        std::get<1>(betas));
    TORCH_CHECK(
        defaults.weight_decay() >= 0,  // 检查权重衰减参数的合法性
        "Invalid weight_decay value: ",
        defaults.weight_decay());
  }
  explicit AdamW(std::vector<Tensor> params, AdamWOptions defaults = {})  // 显式构造函数，传入参数和选项
      : AdamW({OptimizerParamGroup(std::move(params))}, defaults) {}  // 调用其他构造函数进行初始化

  torch::Tensor step(LossClosure closure = nullptr) override;  // 步进函数声明
  void save(serialize::OutputArchive& archive) const override;  // 保存函数声明
  void load(serialize::InputArchive& archive) override;  // 加载函数声明

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {  // 序列化静态函数模板声明
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(AdamW);  // 调用宏进行序列化
  }
};
} // namespace optim
} // namespace torch
```