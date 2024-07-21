# `.\pytorch\torch\csrc\jit\mobile\train\optim\sgd.h`

```py
#pragma once

#include <torch/arg.h> // 包含 torch 库中的 arg 模块
#include <torch/nn/module.h> // 包含 torch 库中的 nn 模块
#include <torch/serialize/archive.h> // 包含 torch 库中的 serialize 模块
#include <torch/types.h> // 包含 torch 库中的 types 模块

#include <cstddef> // 包含标准库中的 cstddef 头文件
#include <utility> // 包含标准库中的 utility 头文件
#include <vector> // 包含标准库中的 vector 头文件

namespace torch { // 命名空间 torch
namespace jit { // 命名空间 jit
namespace mobile { // 命名空间 mobile

class SGDParamState { // 定义类 SGDParamState
  TORCH_ARG(torch::Tensor, momentum_buffer); // 定义一个名为 momentum_buffer 的 torch::Tensor 类型参数

 public:
  std::unique_ptr<SGDParamState> clone() const { // 定义 clone 方法，返回一个 SGDParamState 类型的唯一指针
    return std::make_unique<SGDParamState>( // 创建一个新的 SGDParamState 对象
        static_cast<const SGDParamState&>(*this)); // 使用当前对象进行初始化
  }
  friend bool operator==(const SGDParamState& lhs, const SGDParamState& rhs); // 声明友元函数，用于比较两个 SGDParamState 对象是否相等
};

struct TORCH_API SGDOptions { // 定义结构体 SGDOptions
  /* implicit */ SGDOptions(double lr); // 隐式构造函数，接受一个 double 类型参数 lr
  TORCH_ARG(double, lr); // 定义一个名为 lr 的 double 类型参数
  TORCH_ARG(double, momentum) = 0; // 定义一个名为 momentum 的 double 类型参数，默认值为 0
  TORCH_ARG(double, dampening) = 0; // 定义一个名为 dampening 的 double 类型参数，默认值为 0
  TORCH_ARG(double, weight_decay) = 0; // 定义一个名为 weight_decay 的 double 类型参数，默认值为 0
  TORCH_ARG(bool, nesterov) = false; // 定义一个名为 nesterov 的 bool 类型参数，默认值为 false

 public:
  std::unique_ptr<SGDOptions> clone() const { // 定义 clone 方法，返回一个 SGDOptions 类型的唯一指针
    return std::make_unique<SGDOptions>(static_cast<const SGDOptions&>(*this)); // 创建一个新的 SGDOptions 对象，使用当前对象进行初始化
  }
  TORCH_API friend bool operator==( // 声明友元函数，用于比较两个 SGDOptions 对象是否相等
      const SGDOptions& lhs,
      const SGDOptions& rhs);
};

/// Stores parameters in the param_group and stores a pointer to the SGDOptions
class TORCH_API SGDParamGroup { // 定义类 SGDParamGroup
 public:
  // NOTE: In order to store `SGDParamGroup` in a `std::vector`, it has to be
  // copy-constructible.
  SGDParamGroup(const SGDParamGroup& param_group) // 定义拷贝构造函数
      : params_(param_group.params()), // 初始化 params_ 成员变量
        options_( // 初始化 options_ 成员变量
            param_group.has_options() ? param_group.options().clone()
                                      : nullptr) {} // 根据是否有 options 来选择是否克隆 options

  SGDParamGroup& operator=(const SGDParamGroup& param_group) { // 定义赋值运算符重载函数
    this->params_ = param_group.params(); // 将参数组的参数赋值给当前对象的参数
    this->options_ = // 根据是否有 options 来选择是否克隆 options
        param_group.has_options() ? param_group.options().clone() : nullptr;
    return *this; // 返回当前对象
  }
  /* implicit */ SGDParamGroup(std::vector<Tensor> params) // 隐式构造函数，接受一个 Tensor 类型的 vector 参数
      : params_(std::move(params)) {} // 初始化 params_ 成员变量

  SGDParamGroup(std::vector<Tensor> params, std::unique_ptr<SGDOptions> options) // 构造函数，接受一个 Tensor 类型的 vector 参数和一个唯一指针参数
      : params_(std::move(params)), options_(std::move(options)) {} // 初始化 params_ 和 options_ 成员变量

  bool has_options() const; // 声明 has_options 方法，用于检查是否有 options
  SGDOptions& options(); // 声明 options 方法，返回 options 成员变量的引用
  const SGDOptions& options() const; // 声明 options 方法，返回 options 成员变量的常量引用
  void set_options(std::unique_ptr<SGDOptions> options); // 声明 set_options 方法，设置 options 成员变量
  std::vector<Tensor>& params(); // 声明 params 方法，返回 params 成员变量的引用
  const std::vector<Tensor>& params() const; // 声明 params 方法，返回 params 成员变量的常量引用

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<Tensor> params_; // 定义一个 Tensor 类型的 vector 成员变量 params_
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::unique_ptr<SGDOptions> options_; // 定义一个唯一指针类型的成员变量 options_
};

class TORCH_API SGD { // 定义类 SGD
 public:
  explicit SGD( // 显式构造函数
      const std::vector<torch::jit::mobile::SGDParamGroup>& param_groups, // 接受一个 SGDParamGroup 类型的 vector 参数
      SGDOptions defaults) // 接受一个 SGDOptions 参数
      : defaults_(std::make_unique<SGDOptions>(defaults)) { // 初始化 defaults_ 成员变量
    for (const auto& param_group : param_groups) { // 遍历 param_groups
      add_param_group(param_group); // 添加参数组
    }
    TORCH_CHECK(defaults.lr() >= 0, "Invalid learning rate: ", defaults.lr()); // 检查学习率是否大于等于 0
    TORCH_CHECK( // 检查动量值是否大于等于 0
        defaults.momentum() >= 0,
        "Invalid momentum value: ",
        defaults.momentum());
    // 检查默认的权重衰减值是否大于等于零，如果小于零则抛出异常
    TORCH_CHECK(
        defaults.weight_decay() >= 0,
        "Invalid weight_decay value: ",
        defaults.weight_decay());

    // 如果使用 Nesterov 动量，则要求动量大于零且阻尼为零，否则抛出异常
    TORCH_CHECK(
        !defaults.nesterov() ||
            (defaults.momentum() > 0 && defaults.dampening() == 0),
        "Nesterov momentum requires a momentum and zero dampening");
  }

  // 显式构造函数，接受一组参数和默认选项作为参数
  explicit SGD(std::vector<Tensor> params, SGDOptions defaults)
      : SGD({SGDParamGroup(std::move(params))}, defaults) {}

  /// 将给定的 param_group 添加到优化器的 param_group 列表中
  void add_param_group(const SGDParamGroup& param_group);

  // 默认析构函数
  ~SGD() = default;

  // 定义 LossClosure 类型为返回 Tensor 的函数闭包
  using LossClosure = std::function<Tensor()>;
  /// 步进函数，执行优化步骤，接受一个返回 Tensor 的闭包函数作为参数
  torch::Tensor step(const LossClosure& closure = nullptr);

  /// 清空所有参数的梯度
  void zero_grad();

 protected:
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // 优化器的参数组列表
  std::vector<SGDParamGroup> param_groups_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // 状态映射，映射指针到参数状态对象的唯一指针
  ska::flat_hash_map<void*, std::unique_ptr<SGDParamState>> state_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // 默认的 SGDOptions 对象的唯一指针
  std::unique_ptr<SGDOptions> defaults_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // 参数列表
  std::vector<Tensor> params_;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  // SGDOptions 对象的唯一指针，用于选择性地配置参数
  std::unique_ptr<SGDOptions> options_;
};
// 结束 torch 命名空间
} // namespace mobile
// 结束 mobile 命名空间
} // namespace jit
// 结束 jit 命名空间
} // namespace torch
```