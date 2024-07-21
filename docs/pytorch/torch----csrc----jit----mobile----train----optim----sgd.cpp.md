# `.\pytorch\torch\csrc\jit\mobile\train\optim\sgd.cpp`

```py
// 包含了SGD优化器的头文件
#include <torch/csrc/jit/mobile/train/optim/sgd.h>

// 包含了Tensor类型相关的头文件
#include <torch/types.h>
// 包含了torch库的实用工具函数的头文件
#include <torch/utils.h>

// 包含了ATen张量库的头文件
#include <ATen/ATen.h>

// 包含了标准库中的functional头文件，用于支持函数对象
#include <functional>

// 定义了torch命名空间
namespace torch {
// 定义了jit命名空间，用于Just In Time编译
namespace jit {
// 定义了mobile命名空间，用于移动端相关功能
namespace mobile {

// 返回是否存在选项的布尔值，判断是否有配置选项
bool SGDParamGroup::has_options() const {
  return options_ != nullptr;
}

// 返回参数组的SGD选项对象的引用
SGDOptions& SGDParamGroup::options() {
  TORCH_CHECK(has_options());
  return *options_.get();
}

// 返回参数组的SGD选项对象的常量引用
const SGDOptions& SGDParamGroup::options() const {
  TORCH_CHECK(has_options());
  return *options_.get();
}

// 设置参数组的SGD选项对象
void SGDParamGroup::set_options(std::unique_ptr<SGDOptions> options) {
  options_ = std::move(options);
}

// 返回参数组的张量向量的引用
std::vector<Tensor>& SGDParamGroup::params() {
  return params_;
}

// 返回参数组的张量向量的常量引用
const std::vector<Tensor>& SGDParamGroup::params() const {
  return params_;
}

// SGDOptions类的构造函数，初始化学习率lr_
SGDOptions::SGDOptions(double lr) : lr_(lr) {}

// 判断两个SGDOptions对象是否相等的全局函数
bool operator==(const SGDOptions& lhs, const SGDOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.momentum() == rhs.momentum()) &&
      (lhs.dampening() == rhs.dampening()) &&
      (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.nesterov() == rhs.nesterov());
}

// 判断两个SGDParamState对象是否相等的全局函数，仅比较momentum_buffer张量是否相等
bool operator==(const SGDParamState& lhs, const SGDParamState& rhs) {
  return torch::equal(lhs.momentum_buffer(), rhs.momentum_buffer());
}

// 向SGD优化器添加参数组的函数
void SGD::add_param_group(const SGDParamGroup& param_group) {
  // 检查参数组中的每个张量是否为叶子节点，不能优化非叶子张量
  for (const auto& param : param_group.params()) {
    TORCH_CHECK(param.is_leaf(), "can't optimize a non-leaf Tensor");
  }
  // 内部断言检查默认选项不为空
  TORCH_INTERNAL_ASSERT(defaults_ != nullptr);
  // 复制参数组并设置默认选项
  SGDParamGroup param_group_(param_group.params());
  if (!param_group.has_options()) {
    param_group_.set_options(defaults_->clone());
  } else {
    param_group_.set_options(param_group.options().clone());
  }
  // 检查每个参数是否只出现在一个参数组中
  for (const auto& p : param_group_.params()) {
    TORCH_CHECK(
        state_.count(p.unsafeGetTensorImpl()) == 0,
        "some parameters appear in more than one parameter group");
  }
  // 将参数组移动到param_groups_中
  param_groups_.emplace_back(std::move(param_group_));
}

// 将所有参数的梯度置零
void SGD::zero_grad() {
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      if (p.grad().defined()) {
        p.grad().detach_();
        p.grad().zero_();
      }
    }
  }
}

// 执行一步优化更新
Tensor SGD::step(const LossClosure& closure) {
  // 禁止梯度计算的上下文管理器
  NoGradGuard no_grad;
  Tensor loss = {};
  // 如果闭包不为空，开启自动梯度计算模式并计算损失
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  // 遍历每个参数组进行优化步骤
  for (auto& group : param_groups_) {
    auto& options = static_cast<SGDOptions&>(group.options());
    auto weight_decay = options.weight_decay();
    auto momentum = options.momentum();
    auto dampening = options.dampening();
    auto nesterov = options.nesterov();
    // 遍历参数组中的每个参数 p
    for (auto& p : group.params()) {
        // 如果当前参数的梯度未定义，则跳过该参数
        if (!p.grad().defined()) {
            continue;
        }
        // 获取当前参数的梯度数据
        auto d_p = p.grad().data();
        // 如果设置了权重衰减，则在梯度上加上对应的权重衰减项
        if (weight_decay != 0) {
            d_p = d_p.add(p.data(), weight_decay);
        }
        // 如果设置了动量，则进行动量更新
        if (momentum != 0) {
            // 定义一个缓冲区 buf
            Tensor buf;
            // 查找当前参数的状态
            auto param_state = state_.find(p.unsafeGetTensorImpl());
            // 如果状态中没有当前参数的记录
            if (param_state == state_.end()) {
                // 克隆当前梯度 d_p，并分离其计算图
                buf = torch::clone(d_p).detach();
                // 创建一个新的 SGDParamState 状态对象
                auto state = std::make_unique<SGDParamState>();
                // 设置动量缓冲区为 buf
                state->momentum_buffer(buf);
                // 将参数及其状态存入状态字典 state_
                state_[p.unsafeGetTensorImpl()] = std::move(state);
            } else {
                // 如果状态中已经有当前参数的记录，则使用已有的动量缓冲区 buf
                buf = static_cast<SGDParamState&>(*param_state->second)
                          .momentum_buffer();
                // 更新动量缓冲区 buf，根据动量 momentum 和阻尼 dampening
                buf.mul_(momentum).add_(d_p, 1 - dampening);
            }
            // 如果使用 Nesterov 动量，则在梯度上加上动量缓冲区 buf
            if (nesterov) {
                d_p = d_p.add(buf, momentum);
            } else {
                // 否则直接将梯度设置为 buf
                d_p = buf;
            }
        }
        // 对当前参数的值应用梯度下降步骤，乘以负的学习率 options.lr()
        p.data().add_(d_p, -1 * options.lr());
    }
}
// 返回损失值 loss
return loss;
} // 关闭 torch 命名空间
} // 关闭 jit 命名空间
} // 关闭 mobile 命名空间
```