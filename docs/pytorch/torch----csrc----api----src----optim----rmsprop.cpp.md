# `.\pytorch\torch\csrc\api\src\optim\rmsprop.cpp`

```
/// 包含 RMSprop 优化器的实现
#include <torch/optim/rmsprop.h>

/// 包含变量自动求导相关头文件
#include <torch/csrc/autograd/variable.h>
/// 包含序列化存档相关头文件
#include <torch/serialize/archive.h>
/// 包含 Torch 工具函数相关头文件
#include <torch/utils.h>

/// 包含 ATen 库的核心头文件
#include <ATen/ATen.h>
/// 包含 C10 库的工具函数相关头文件
#include <c10/util/irange.h>

/// 包含 C++ 标准库的功能函数
#include <functional>

/// Torch 命名空间
namespace torch {
/// 优化器命名空间
namespace optim {

/// RMSpropOptions 类的构造函数，初始化学习率 lr_
RMSpropOptions::RMSpropOptions(double lr) : lr_(lr) {}

/// 判断两个 RMSpropOptions 对象是否相等的重载运算符
bool operator==(const RMSpropOptions& lhs, const RMSpropOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.alpha() == rhs.alpha()) &&
      (lhs.eps() == rhs.eps()) && (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.momentum() == rhs.momentum()) && (lhs.centered() == rhs.centered());
}

/// 将 RMSpropOptions 对象序列化为输出存档
void RMSpropOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(alpha);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(centered);
}

/// 从输入存档反序列化 RMSpropOptions 对象
void RMSpropOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, alpha);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, centered);
}

/// 获取当前学习率 lr_
double RMSpropOptions::get_lr() const {
  return lr();
}

/// 设置当前学习率 lr_
void RMSpropOptions::set_lr(const double lr) {
  this->lr(lr);
}

/// 判断两个 RMSpropParamState 对象是否相等的重载运算符
bool operator==(const RMSpropParamState& lhs, const RMSpropParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.square_avg(), rhs.square_avg()) &&
      torch::equal_if_defined(lhs.momentum_buffer(), rhs.momentum_buffer()) &&
      torch::equal_if_defined(lhs.grad_avg(), rhs.grad_avg());
}

/// 将 RMSpropParamState 对象序列化为输出存档
void RMSpropParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(square_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum_buffer);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(grad_avg);
}

/// 从输入存档反序列化 RMSpropParamState 对象
void RMSpropParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, square_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, momentum_buffer);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, grad_avg);
}

/// RMSprop 类的步骤函数实现，根据损失函数进行参数更新
/// 闭包函数 closure 可选地计算损失
Tensor RMSprop::step(LossClosure closure) {
  /// 禁用梯度追踪
  NoGradGuard no_grad;
  /// 初始化损失值
  Tensor loss = {};
  /// 如果闭包函数不为空，则在允许梯度模式下计算损失值
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  /// 遍历每个参数组
  for (auto& group : param_groups_) {
    // 对于每个参数 p 在优化器组 group 中的参数集合中
    for (auto& p : group.params()) {
      // 如果梯度未定义，跳过该参数
      if (!p.grad().defined()) {
        continue;
      }
      // 获取参数的梯度
      auto grad = p.grad();
      // 检查梯度是否是稀疏的，RMSprop 不支持稀疏梯度
      TORCH_CHECK(
          !grad.is_sparse(), "RMSprop does not support sparse gradients");
      // 查找参数状态 state_
      auto param_state = state_.find(p.unsafeGetTensorImpl());
      // 获取 RMSpropOptions
      auto& options = static_cast<RMSpropOptions&>(group.options());

      // 状态初始化
      if (param_state == state_.end()) {
        // 如果参数状态不存在，创建新的状态对象
        auto state = std::make_unique<RMSpropParamState>();
        // 初始化步数为 0
        state->step(0);
        // 初始化平方平均值为和参数 p 相同形状的全零张量
        state->square_avg(torch::zeros_like(p, MemoryFormat::Preserve));
        // 如果设置了动量选项，初始化动量缓存为全零张量
        if (options.momentum() > 0) {
          state->momentum_buffer(torch::zeros_like(p, MemoryFormat::Preserve));
        }
        // 如果设置了 centered 选项，初始化梯度平均值为全零张量
        if (options.centered()) {
          state->grad_avg(torch::zeros_like(p, MemoryFormat::Preserve));
        }
        // 将参数 p 的状态存入 state_
        state_[p.unsafeGetTensorImpl()] = std::move(state);
      }

      // 获取参数 p 的状态对象
      auto& state =
          static_cast<RMSpropParamState&>(*state_[p.unsafeGetTensorImpl()]);
      // 获取参数的平方平均值
      auto& square_avg = state.square_avg();
      // 获取学习率衰减因子 alpha
      auto alpha = options.alpha();

      // 更新步数
      state.step(state.step() + 1);

      // 如果设置了权重衰减选项，对梯度 grad 进行权重衰减
      if (options.weight_decay() != 0) {
        grad = grad.add(p, options.weight_decay());
      }

      // 更新平方平均值
      square_avg.mul_(alpha).addcmul_(grad, grad, 1 - alpha);

      // 定义平均值 avg
      Tensor avg;
      // 如果设置了 centered 选项
      if (options.centered()) {
        // 获取梯度平均值
        auto& grad_avg = state.grad_avg();
        // 更新梯度平均值
        grad_avg.mul_(alpha).add_(grad, 1 - alpha);
        // 计算 avg，包括平方平均值和梯度平均值的影响
        avg = square_avg.addcmul(grad_avg, grad_avg, -1)
                  .sqrt_()
                  .add_(options.eps());
      } else {
        // 计算 avg，仅包括平方平均值的影响
        avg = square_avg.sqrt().add_(options.eps());
      }

      // 如果设置了动量选项
      if (options.momentum() > 0) {
        // 获取动量缓存
        auto& buf = state.momentum_buffer();
        // 更新动量缓存
        buf.mul_(options.momentum()).addcdiv_(grad, avg);
        // 需要避免对参数进行版本跟踪
        p.add_(buf, -options.lr());
      } else {
        // 需要避免对参数进行版本跟踪
        p.addcdiv_(grad, avg, -options.lr());
      }
    }
  }
  // 返回损失值
  return loss;
}

void RMSprop::save(serialize::OutputArchive& archive) const {
  // 将当前 RMSprop 对象序列化保存到输出存档中
  serialize(*this, archive);
}

void RMSprop::load(serialize::InputArchive& archive) {
  // 创建一个 IValue 对象，用于存储 pytorch_version 的值
  IValue pytorch_version;
  // 尝试从存档中读取名为 "pytorch_version" 的值到 pytorch_version 变量
  if (archive.try_read("pytorch_version", pytorch_version)) {
    // 如果成功读取到 pytorch_version，则进行反序列化操作
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    // 输出警告信息，指出正在使用旧的序列化格式
    TORCH_WARN(
        "Your serialized RMSprop optimizer is still using the old serialization format. "
        "The step value in state will be set to 0 because the old RMSprop optimizer didn't track the step value."
        "You should re-save your RMSprop optimizer to use the new serialization format.");
    // 创建三个 Tensor 向量，用于存储 square_average_buffers、momentum_buffers 和 grad_average_buffers
    std::vector<Tensor> square_average_buffers;
    std::vector<Tensor> momentum_buffers;
    std::vector<Tensor> grad_average_buffers;
    // 序列化存储 square_average_buffers 到存档中
    torch::optim::serialize(
        archive, "square_average_buffers", square_average_buffers);
    // 序列化存储 momentum_buffers 到存档中
    torch::optim::serialize(archive, "momentum_buffers", momentum_buffers);
    // 序列化存储 grad_average_buffers 到存档中
    torch::optim::serialize(
        archive, "grad_average_buffers", grad_average_buffers);
    // 因为在版本 1.5.0 之前不存在 param_groups，假设所有张量现在都在一个 param_group 中
    // 从 param_groups_ 中获取第一个 param_group 中的参数张量列表
    std::vector<Tensor> params = param_groups_.at(0).params();
    // 遍历 square_average_buffers 的索引范围
    for (const auto idx : c10::irange(square_average_buffers.size())) {
      // 创建一个 RMSpropParamState 的智能指针 state
      auto state = std::make_unique<RMSpropParamState>();
      // 设置 state 的 square_avg 属性为 square_average_buffers[idx]
      state->square_avg(square_average_buffers[idx]);
      // 如果 idx 小于 momentum_buffers 的大小，则设置 state 的 momentum_buffer 属性
      if (idx < momentum_buffers.size()) {
        state->momentum_buffer(momentum_buffers.at(idx));
      }
      // 如果 idx 小于 grad_average_buffers 的大小，则设置 state 的 grad_avg 属性
      if (idx < grad_average_buffers.size()) {
        state->grad_avg(grad_average_buffers.at(idx));
      }
      // 将 state 对象存储到 state_ 映射中，键为 params[idx] 的不安全张量实现
      state_[params[idx].unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
} // namespace optim
} // namespace torch
```