# `.\pytorch\torch\csrc\api\include\torch\optim\sgd.h`

```
#pragma once

#include <torch/nn/module.h>  // 包含 Torch 的神经网络模块相关头文件
#include <torch/optim/optimizer.h>  // 包含 Torch 的优化器相关头文件
#include <torch/optim/serialize.h>  // 包含 Torch 的优化器序列化相关头文件
#include <torch/serialize/archive.h>  // 包含 Torch 的存档处理相关头文件
#include <torch/types.h>  // 包含 Torch 的数据类型定义

#include <cstddef>  // 包含标准库头文件，定义了 size_t 类型等
#include <utility>  // 包含标准库头文件，定义了 std::move 等实用函数
#include <vector>  // 包含标准库头文件，定义了 std::vector 容器

namespace torch {
namespace serialize {
class OutputArchive;  // 声明 Torch 序列化命名空间中的输出存档类
class InputArchive;   // 声明 Torch 序列化命名空间中的输入存档类
} // namespace serialize
} // namespace torch

namespace torch {
namespace optim {

// SGDOptions 结构体继承自 OptimizerCloneableOptions 类，定义了 SGD 优化器的选项
struct TORCH_API SGDOptions : public OptimizerCloneableOptions<SGDOptions> {
  SGDOptions(double lr);  // 构造函数，接受学习率 lr 参数
  TORCH_ARG(double, lr);  // 学习率 lr 成员变量
  TORCH_ARG(double, momentum) = 0;  // 冲量 momentum 成员变量，默认值为 0
  TORCH_ARG(double, dampening) = 0;  // 阻尼 dampening 成员变量，默认值为 0
  TORCH_ARG(double, weight_decay) = 0;  // 权重衰减 weight_decay 成员变量，默认值为 0
  TORCH_ARG(bool, nesterov) = false;  // Nesterov 动量 nesterov 成员变量，默认值为 false

 public:
  void serialize(torch::serialize::InputArchive& archive) override;  // 序列化函数，从输入存档中反序列化对象
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 序列化函数，将对象序列化到输出存档中
  TORCH_API friend bool operator==(  // 友元函数，比较两个 SGDOptions 对象是否相等
      const SGDOptions& lhs,
      const SGDOptions& rhs);
  double get_lr() const override;  // 获取学习率 lr 的方法
  void set_lr(const double lr) override;  // 设置学习率 lr 的方法
};

// SGDParamState 结构体继承自 OptimizerCloneableParamState 类，定义了 SGD 优化器的参数状态
struct TORCH_API SGDParamState
    : public OptimizerCloneableParamState<SGDParamState> {
  TORCH_ARG(torch::Tensor, momentum_buffer);  // 冲量缓存 momentum_buffer 成员变量

 public:
  void serialize(torch::serialize::InputArchive& archive) override;  // 序列化函数，从输入存档中反序列化对象
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 序列化函数，将对象序列化到输出存档中
  TORCH_API friend bool operator==(  // 友元函数，比较两个 SGDParamState 对象是否相等
      const SGDParamState& lhs,
      const SGDParamState& rhs);
};

// SGD 类继承自 Optimizer 类，实现了随机梯度下降优化器
class TORCH_API SGD : public Optimizer {
 public:
  explicit SGD(
      std::vector<OptimizerParamGroup> param_groups,  // 构造函数，接受优化器参数组和默认选项
      SGDOptions defaults)
      : Optimizer(
            std::move(param_groups),  // 调用 Optimizer 类的构造函数，传递参数组
            std::make_unique<SGDOptions>(defaults)) {  // 使用默认选项构造 SGDOptions 对象
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());  // 检查学习率是否合法
    TORCH_CHECK(
        defaults.momentum() >= 0,  // 检查动量值是否合法
        "Invalid momentum value: ",
        defaults.momentum());
    TORCH_CHECK(
        defaults.weight_decay() >= 0,  // 检查权重衰减值是否合法
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(
        !defaults.nesterov() ||  // 如果使用 Nesterov 动量，则要求同时设置动量且阻尼为 0
            (defaults.momentum() > 0 && defaults.dampening() == 0),
        "Nesterov momentum requires a momentum and zero dampening");
  }

  explicit SGD(std::vector<Tensor> params, SGDOptions defaults)  // 另一个构造函数，接受参数列表和默认选项
      : SGD({OptimizerParamGroup(std::move(params))}, defaults) {}  // 调用上述构造函数进行初始化

  torch::Tensor step(LossClosure closure = nullptr) override;  // 实现 Optimizer 类中的 step 方法

  void save(serialize::OutputArchive& archive) const override;  // 保存模型状态到输出存档中的方法
  void load(serialize::InputArchive& archive) override;  // 从输入存档中加载模型状态的方法

 private:
  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {  // 静态模板函数，用于序列化对象
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(SGD);  // 调用宏，处理 SGD 的序列化
  }
};
} // namespace optim
} // namespace torch
```