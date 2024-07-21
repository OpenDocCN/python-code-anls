# `.\pytorch\torch\csrc\api\include\torch\optim\adagrad.h`

```py
#pragma once

#include <torch/nn/pimpl.h>  // 包含神经网络模块的私有实现头文件
#include <torch/optim/optimizer.h>  // 包含优化器的头文件
#include <torch/optim/serialize.h>  // 包含优化器序列化相关头文件
#include <torch/serialize/archive.h>  // 包含存档相关头文件
#include <torch/types.h>  // 包含 Torch 的数据类型定义

#include <utility>  // 包含实用工具，例如 std::move
#include <vector>  // 包含向量容器

namespace torch {
namespace serialize {
class OutputArchive;  // 输出存档类声明
class InputArchive;  // 输入存档类声明
}  // namespace serialize
}  // namespace torch

namespace torch {
namespace optim {

struct TORCH_API AdagradOptions  // Adagrad 优化器选项结构体声明
    : public OptimizerCloneableOptions<AdagradOptions> {  // 继承自克隆优化器选项模板
  AdagradOptions(double lr = 1e-2);  // 构造函数声明，默认学习率为 0.01
  TORCH_ARG(double, lr) = 1e-2;  // 学习率参数，默认值为 0.01
  TORCH_ARG(double, lr_decay) = 0;  // 学习率衰减参数，默认值为 0
  TORCH_ARG(double, weight_decay) = 0;  // 权重衰减参数，默认值为 0
  TORCH_ARG(double, initial_accumulator_value) = 0;  // 初始累加器值，默认值为 0
  TORCH_ARG(double, eps) = 1e-10;  // 精度值，默认值为 1e-10

 public:
  void serialize(torch::serialize::InputArchive& archive) override;  // 输入存档序列化函数声明
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 输出存档序列化函数声明
  TORCH_API friend bool operator==(  // 相等比较运算符重载声明
      const AdagradOptions& lhs,
      const AdagradOptions& rhs);
  double get_lr() const override;  // 获取学习率函数声明
  void set_lr(const double lr) override;  // 设置学习率函数声明
};

struct TORCH_API AdagradParamState  // Adagrad 参数状态结构体声明
    : public OptimizerCloneableParamState<AdagradParamState> {  // 继承自克隆优化器参数状态模板
  TORCH_ARG(torch::Tensor, sum);  // Tensor 类型 sum 参数声明
  TORCH_ARG(int64_t, step) = 0;  // 步数参数声明，默认值为 0

 public:
  AdagradParamState() = default;  // 默认构造函数声明
  AdagradParamState(const AdagradParamState&) = default;  // 拷贝构造函数声明
  AdagradParamState& operator=(const AdagradParamState&) = default;  // 赋值运算符重载声明
  AdagradParamState(AdagradParamState&&) noexcept = default;  // 移动构造函数声明
  AdagradParamState& operator=(AdagradParamState&&) noexcept = default;  // 移动赋值运算符重载声明
  void serialize(torch::serialize::InputArchive& archive) override;  // 输入存档序列化函数声明
  void serialize(torch::serialize::OutputArchive& archive) const override;  // 输出存档序列化函数声明
  TORCH_API friend bool operator==(  // 相等比较运算符重载声明
      const AdagradParamState& lhs,
      const AdagradParamState& rhs);
};

class TORCH_API Adagrad : public Optimizer {  // Adagrad 类声明，继承自优化器类
 public:
  explicit Adagrad(  // 显式构造函数声明
      std::vector<OptimizerParamGroup> param_groups,  // 参数组向量声明
      AdagradOptions defaults = {})  // AdagradOptions 默认参数声明
      : Optimizer(  // 初始化列表开始
            std::move(param_groups),  // 移动参数组向量
            std::make_unique<AdagradOptions>(defaults)) {  // 创建 AdagradOptions 的唯一指针
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr());  // 检查学习率是否有效
    TORCH_CHECK(  // 检查学习率衰减值是否有效
        defaults.lr_decay() >= 0,
        "Invalid lr_decay value: ",
        defaults.lr_decay());
    TORCH_CHECK(  // 检查权重衰减值是否有效
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());
    TORCH_CHECK(  // 检查初始累加器值是否有效
        defaults.initial_accumulator_value() >= 0,
        "Invalid initial_accumulator_value value: ",
        defaults.initial_accumulator_value());
    TORCH_CHECK(defaults.eps() >= 0, "Invalid epsilon value: ", defaults.eps());  // 检查精度值是否有效

    for (const auto& group : param_groups_) {  // 遍历参数组向量
      for (const auto& p : group.params()) {  // 遍历每个参数组的参数
        auto state = std::make_unique<AdagradParamState>();  // 创建 AdagradParamState 的唯一指针
        state->step(0);  // 设置步数为 0
        state->sum(torch::full_like(  // 设置 sum 参数为与 p 相同格式的全局变量
            p.data(),
            defaults.initial_accumulator_value(),
            at::MemoryFormat::Preserve));
        state_[p.unsafeGetTensorImpl()] = std::move(state);  // 移动 state 指针到 state_ 结构中
      }
  }
}



  }



  explicit Adagrad(std::vector<Tensor> params, AdagradOptions defaults = {})
      : Adagrad({OptimizerParamGroup(std::move(params))}, defaults) {}



  torch::Tensor step(LossClosure closure = nullptr) override;



  void save(serialize::OutputArchive& archive) const override;



  void load(serialize::InputArchive& archive) override;



 private:



  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adagrad);
  }


注释：


  }


1. `}`：结束了前面的函数或代码块。
2. `}`：结束了前面的代码块。
3. `explicit Adagrad(std::vector<Tensor> params, AdagradOptions defaults = {})`：
   构造函数声明，接受一个参数向量和一个可选的默认参数对象。
4. `: Adagrad({OptimizerParamGroup(std::move(params))}, defaults) {}`：
   使用参数初始化列表调用另一个构造函数 `Adagrad`。
5. `torch::Tensor step(LossClosure closure = nullptr) override;`：
   声明了一个虚函数 `step`，返回一个 `torch::Tensor`，可接受一个可调用对象作为参数。
6. `void save(serialize::OutputArchive& archive) const override;`：
   声明了一个虚函数 `save`，没有返回值，接受一个 `serialize::OutputArchive` 的引用参数。
7. `void load(serialize::InputArchive& archive) override;`：
   声明了一个虚函数 `load`，没有返回值，接受一个 `serialize::InputArchive` 的引用参数。
8. `private:`：以下的成员都是私有的。
9. 


  template <typename Self, typename Archive>
  static void serialize(Self& self, Archive& archive) {
    _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adagrad);
  }


注释：

  }


- `template <typename Self, typename Archive>`：模板声明，接受两个类型参数 `Self` 和 `Archive`。
- `static void serialize(Self& self, Archive& archive)`：静态成员函数 `serialize` 的定义，接受一个 `Self` 类型引用和一个 `Archive` 类型引用。
- `{ _TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG(Adagrad); }`：调用宏或函数 `_TORCH_OPTIM_SERIALIZE_WITH_TEMPLATE_ARG`，传入模板参数 `Adagrad`。
};
} // namespace optim
} // namespace torch
```