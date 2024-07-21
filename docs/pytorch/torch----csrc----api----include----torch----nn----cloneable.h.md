# `.\pytorch\torch\csrc\api\include\torch\nn\cloneable.h`

```
// 预处理指令，确保本文件在编译时只包含一次
#pragma once

// 包含 Torch 的模块定义和类型定义
#include <torch/nn/module.h>
#include <torch/types.h>
#include <torch/utils.h>

// 包含 C10 库的相关头文件
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>

// 包含标准库的头文件
#include <memory>
#include <utility>

// Torch 的命名空间
namespace torch {
// Torch 的神经网络模块命名空间
namespace nn {

/// `Cloneable` 类模板从 `Module` 类继承，通过 CRTP（Curiously Recurring Template Pattern）
/// 获取子类的静态类型信息，并实现 `clone()` 方法。
/// `clone()` 方法在基类 `Module` 中定义，但不知道具体的子类类型，因此需要在子类中调用。
template <typename Derived>
// NOLINTNEXTLINE(bugprone-exception-escape)
class Cloneable : public Module {
 public:
  using Module::Module;

  /// `reset()` 方法在子类中定义，用于初始化所有具有引用语义的成员，包括参数、缓冲区和子模块。
  virtual void reset() = 0;

  /// 实现对 `Module` 的递归 "深复制"，确保克隆后的模块的所有参数和子模块都与原始模块不同。
  std::shared_ptr<Module> clone(
      const optional<Device>& device = nullopt) const override {
    // 使用 `NoGradGuard` 禁用梯度计算
    NoGradGuard no_grad;

    // 获取当前对象的具体子类引用
    const auto& self = static_cast<const Derived&>(*this);
    // 创建当前对象的副本，并转换为共享指针
    auto copy = std::make_shared<Derived>(self);

    // 清空副本的参数、缓冲区和子模块列表
    copy->parameters_.clear();
    copy->buffers_.clear();
    copy->children_.clear();

    // 调用副本的 `reset()` 方法，初始化其成员
    copy->reset();

    // 检查克隆后的参数数量与原始模块是否一致
    TORCH_CHECK(
        copy->parameters_.size() == parameters_.size(),
        "The cloned module does not have the same number of "
        "parameters as the original module after calling reset(). "
        "Are you sure you called register_parameter() inside reset() "
        "and not the constructor?");

    // 对每个参数进行复制或设备迁移操作
    for (const auto& parameter : named_parameters(/*recurse=*/false)) {
      auto& tensor = *parameter;
      auto data = device && tensor.device() != *device
          ? tensor.to(*device)
          : autograd::Variable(tensor).clone();
      copy->parameters_[parameter.key()].set_data(data);
    }

    // 检查克隆后的缓冲区数量与原始模块是否一致
    TORCH_CHECK(
        copy->buffers_.size() == buffers_.size(),
        "The cloned module does not have the same number of "
        "buffers as the original module after calling reset(). "
        "Are you sure you called register_buffer() inside reset() "
        "and not the constructor?");

    // 对每个缓冲区进行复制或设备迁移操作
    for (const auto& buffer : named_buffers(/*recurse=*/false)) {
      auto& tensor = *buffer;
      auto data = device && tensor.device() != *device
          ? tensor.to(*device)
          : autograd::Variable(tensor).clone();
      copy->buffers_[buffer.key()].set_data(data);
    }
    # 使用 TORCH_CHECK 断言验证克隆后的模块子节点数量与原始模块相同
    TORCH_CHECK(
        copy->children_.size() == children_.size(),
        "The cloned module does not have the same number of "
        "child modules as the original module after calling reset(). "
        "Are you sure you called register_module() inside reset() "
        "and not the constructor?");
    # 遍历当前模块的每个子模块
    for (const auto& child : children_) {
      # 对克隆模块的对应子模块调用 clone_ 方法进行克隆操作
      copy->children_[child.key()]->clone_(*child.value(), device);
    }
    # 返回克隆后的模块
    return copy;
  }

 private:
  # 通过 clone_ 方法实现模块的深度克隆
  void clone_(Module& other, const optional<Device>& device) final {
    # 使用 dynamic_cast 动态转换类型，确保 `other` 的类型为 `Derived`
    // Here we are *pretty* certain that `other's` type is `Derived` (because it
    // was registered under the same name as `this`), but you never know what
    // crazy things `reset()` does, so `dynamic_cast` just to be safe.
    auto clone = std::dynamic_pointer_cast<Derived>(other.clone(device));
    # 使用 TORCH_CHECK 断言验证克隆结果不为空
    TORCH_CHECK(
        clone != nullptr,
        "Attempted to clone submodule, but it is of a "
        "different type than the submodule it was to be cloned into");
    # 将克隆结果赋值给当前对象，确保类型匹配
    static_cast<Derived&>(*this) = *clone;
  }
};

// 结束 nn 命名空间
} // namespace nn
// 结束 torch 命名空间
} // namespace torch
```