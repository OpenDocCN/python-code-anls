# `.\pytorch\torch\csrc\api\include\torch\optim\optimizer.h`

```
#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/flat_hash_map.h>

#include <torch/arg.h>
#include <torch/csrc/Export.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

// Forward declarations confuse Doxygen
#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace at {
class Tensor;
} // namespace at

namespace torch {
using at::Tensor;
namespace serialize {
class OutputArchive;
class InputArchive;
} // namespace serialize
} // namespace torch
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace torch {
namespace optim {

// 声明 OptimizerParamState 类，用于存储优化器参数状态
class TORCH_API OptimizerParamState {
 public:
  // 默认构造函数
  OptimizerParamState() = default;
  // 拷贝构造函数
  OptimizerParamState(const OptimizerParamState&) = default;
  // 拷贝赋值运算符
  OptimizerParamState& operator=(const OptimizerParamState&) = default;
  // 移动构造函数
  OptimizerParamState(OptimizerParamState&&) noexcept = default;
  // 移动赋值运算符
  OptimizerParamState& operator=(OptimizerParamState&&) noexcept = default;
  // 克隆函数，返回当前对象的克隆
  virtual std::unique_ptr<OptimizerParamState> clone() const;
  // 序列化函数，从输入存档中反序列化对象状态
  virtual void serialize(torch::serialize::InputArchive& archive);
  // 序列化函数，将对象状态序列化到输出存档中
  virtual void serialize(torch::serialize::OutputArchive& archive) const;
  // 虚析构函数，确保在派生类对象销毁时正确释放资源
  virtual ~OptimizerParamState() = default;
};

// 模板类 OptimizerCloneableParamState，继承自 OptimizerParamState，用于支持派生类的克隆
template <typename Derived>
class OptimizerCloneableParamState : public OptimizerParamState {
  // 实现克隆函数，返回当前对象的派生类克隆
  std::unique_ptr<OptimizerParamState> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

// 声明 OptimizerOptions 类，用于存储优化器选项
class TORCH_API OptimizerOptions {
 public:
  // 默认构造函数
  OptimizerOptions() = default;
  // 拷贝构造函数
  OptimizerOptions(const OptimizerOptions&) = default;
  // 拷贝赋值运算符
  OptimizerOptions& operator=(const OptimizerOptions&) = default;
  // 移动构造函数
  OptimizerOptions(OptimizerOptions&&) noexcept = default;
  // 移动赋值运算符
  OptimizerOptions& operator=(OptimizerOptions&&) noexcept = default;
  // 克隆函数，返回当前对象的克隆
  virtual std::unique_ptr<OptimizerOptions> clone() const;
  // 序列化函数，从输入存档中反序列化对象状态
  virtual void serialize(torch::serialize::InputArchive& archive);
  // 序列化函数，将对象状态序列化到输出存档中
  virtual void serialize(torch::serialize::OutputArchive& archive) const;
  // 虚析构函数，确保在派生类对象销毁时正确释放资源
  virtual ~OptimizerOptions() = default;
  // 获取学习率的虚函数，用于派生类提供实现
  virtual double get_lr() const;
  // 设置学习率的虚函数，用于派生类提供实现
  virtual void set_lr(const double lr);
};

// 模板类 OptimizerCloneableOptions，继承自 OptimizerOptions，用于支持派生类的克隆
template <typename Derived>
class OptimizerCloneableOptions : public OptimizerOptions {
 private:
  // 实现克隆函数，返回当前对象的派生类克隆
  std::unique_ptr<OptimizerOptions> clone() const override {
    return std::make_unique<Derived>(static_cast<const Derived&>(*this));
  }
};

/// Stores parameters in the param_group and stores a pointer to the
/// OptimizerOptions
// 在 param_group 中存储参数，并保存指向 OptimizerOptions 的指针
// 定义一个名为 OptimizerParamGroup 的类，用于管理优化器的参数组
class TORCH_API OptimizerParamGroup {
 public:
  // 注意：为了能够将 OptimizerParamGroup 存储在 std::vector 中，必须是可拷贝构造的
  OptimizerParamGroup(const OptimizerParamGroup& param_group)
      : params_(param_group.params()),  // 使用参数组的 params 进行拷贝构造
        options_(
            param_group.has_options() ? param_group.options().clone()  // 如果参数组有选项，则进行克隆
                                      : nullptr) {}  // 否则选项为空指针
  // 构造函数，接受一个 Tensor 类型的向量作为参数
  OptimizerParamGroup(std::vector<Tensor> params)
      : params_(std::move(params)) {}  // 初始化 params_ 成员变量

  // 构造函数，接受一个 Tensor 类型的向量和一个 OptimizerOptions 类型的独占指针作为参数
  OptimizerParamGroup(
      std::vector<Tensor> params,
      std::unique_ptr<OptimizerOptions> options)
      : params_(std::move(params)), options_(std::move(options)) {}  // 初始化 params_ 和 options_ 成员变量

  // 检查是否存在选项对象
  bool has_options() const;

  // 返回选项对象的非常量引用
  OptimizerOptions& options();

  // 返回选项对象的常量引用
  const OptimizerOptions& options() const;

  // 设置选项对象，接受一个 OptimizerOptions 类型的独占指针作为参数
  void set_options(std::unique_ptr<OptimizerOptions> options);

  // 返回参数向量的非常量引用
  std::vector<Tensor>& params();

  // 返回参数向量的常量引用
  const std::vector<Tensor>& params() const;

 protected:
  std::vector<Tensor> params_;  // 存储 Tensor 类型的向量
  std::unique_ptr<OptimizerOptions> options_;  // 独占指针，指向 OptimizerOptions 类型的对象
};

// 定义一个名为 Optimizer 的类，用于实现优化器的相关功能
class TORCH_API Optimizer {
 public:
  // 删除拷贝构造函数，用户应使用 state_dict / load_state_dict API 复制优化器
  Optimizer(const Optimizer& optimizer) = delete;

  // 移动构造函数，默认实现
  Optimizer(Optimizer&& optimizer) = default;

  // 构造函数，接受一个 OptimizerParamGroup 类型的向量和一个 OptimizerOptions 类型的独占指针作为参数
  explicit Optimizer(
      std::vector<OptimizerParamGroup> param_groups,
      std::unique_ptr<OptimizerOptions> defaults)
      : defaults_(std::move(defaults)) {  // 初始化 defaults_ 成员变量
    // 遍历参数组向量中的每个参数组，并添加到优化器中
    for (const auto& param_group : param_groups) {
      add_param_group(param_group);
    /// 结束了一个类的定义
    }
      }
    
    /// 从参数向量构造 `Optimizer` 对象。
    explicit Optimizer(
        std::vector<Tensor> parameters,                /// 参数向量
        std::unique_ptr<OptimizerOptions> defaults)    /// 独占指针指向优化器选项
        : Optimizer(
              {OptimizerParamGroup(std::move(parameters))},  /// 通过参数向量构造包含单个参数组的 `Optimizer`
              std::move(defaults)){};
    
    /// 将给定的 `param_group` 添加到优化器的参数组列表中。
    void add_param_group(const OptimizerParamGroup& param_group);
    
    virtual ~Optimizer() = default;  /// 虚析构函数，默认实现
    
    using LossClosure = std::function<Tensor()>;  /// 定义 `LossClosure` 类型为返回 `Tensor` 的函数
    /// 执行一步优化操作，使用损失函数闭包 `closure` 返回损失值。
    virtual Tensor step(LossClosure closure = nullptr) = 0;
    
    /// 将给定的参数向量添加到优化器的参数列表中。
    void add_parameters(const std::vector<Tensor>& parameters);
    
    /// 将所有参数的梯度归零。
    void zero_grad(bool set_to_none = true);
    
    /// 返回第一个参数组中的参数的常量引用。
    const std::vector<Tensor>& parameters() const noexcept;
    
    /// 返回第一个参数组中的参数的引用。
    std::vector<Tensor>& parameters() noexcept;
    
    /// 返回优化器引用的参数数目。
    size_t size() const noexcept;
    
    /// 返回默认选项的引用。
    OptimizerOptions& defaults() noexcept;
    
    /// 返回默认选项的常量引用。
    const OptimizerOptions& defaults() const noexcept;
    
    /// 返回优化器持有的参数组的引用。
    std::vector<OptimizerParamGroup>& param_groups() noexcept;
    
    /// 返回优化器持有的参数组的常量引用。
    const std::vector<OptimizerParamGroup>& param_groups() const noexcept;
    
    /// 返回优化器持有的状态的引用。
    ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>&
    state() noexcept;
    
    /// 返回优化器持有的状态的常量引用。
    const ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>>& state()
        const noexcept;
    
    /// 将优化器状态序列化到给定的 `archive` 中。
    virtual void save(serialize::OutputArchive& archive) const;
    
    /// 从给定的 `archive` 中反序列化优化器状态。
    virtual void load(serialize::InputArchive& archive);
    
    protected:
    std::vector<OptimizerParamGroup> param_groups_;   /// 优化器的参数组列表
    ska::flat_hash_map<void*, std::unique_ptr<OptimizerParamState>> state_;   /// 优化器持有的状态映射
    std::unique_ptr<OptimizerOptions> defaults_;   /// 独占指针指向默认选项
};

/* How do we decide whether to serialize undefined tensors or
  c10::nullopt values into the output archive?
Answer: we strictly follow the behavior of Python API. To be more specific:

For optimizer options:
a) For undefined tensor: currently no tensor is used as an options argument in
Python API, so we don't need to worry about it now. b) For c10::nullopt value:
we serialize c10::nullopt values into the output archive, to follow the exact
same behavior as Python API.

For optimizer param state:
a) For undefined tensor: in param state, undefined tensor in C++ impl is
equivalent to missing key in Python impl. Since we don't serialize missing keys
in Python API, we skip undefined tensors when serializing the param state. b)
For c10::nullopt value: in param state, c10::nullopt value in C++ impl is
equivalent to missing key in Python impl. Since we don't serialize missing keys
in Python API, we skip c10::nullopt values when serializing the param state. */

/// Serializes an `Optimizer` into an `OutputArchive`.
TORCH_API serialize::OutputArchive& operator<<(
    serialize::OutputArchive& archive,
    const Optimizer& optimizer);



/// Deserializes a `Tensor` from an `InputArchive`.
TORCH_API serialize::InputArchive& operator>>(
    serialize::InputArchive& archive,
    Optimizer& optimizer);

} // namespace optim
} // namespace torch
```