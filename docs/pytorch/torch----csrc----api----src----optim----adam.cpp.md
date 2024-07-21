# `.\pytorch\torch\csrc\api\src\optim\adam.cpp`

```
// 包含 PyTorch 中 Adam 优化器的头文件
#include <torch/optim/adam.h>

// 包含 PyTorch 自动微分变量的头文件
#include <torch/csrc/autograd/variable.h>
// 包含 PyTorch 模块的头文件
#include <torch/nn/module.h>
// 包含 PyTorch 序列化档案的头文件
#include <torch/serialize/archive.h>
// 包含 PyTorch 实用工具的头文件
#include <torch/utils.h>

// 包含 ATen 张量库的头文件
#include <ATen/ATen.h>
// 包含 C10 实用工具的头文件
#include <c10/util/irange.h>

// 包含数学函数的头文件
#include <cmath>
// 包含函数式编程的头文件
#include <functional>

// 定义了 PyTorch 中 Adam 优化器的命名空间 torch::optim
namespace torch {
namespace optim {

// 构造函数 AdamOptions::AdamOptions，初始化学习率 lr_
AdamOptions::AdamOptions(double lr) : lr_(lr) {}

// 判断两个 AdamOptions 对象是否相等的重载运算符
bool operator==(const AdamOptions& lhs, const AdamOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&
      (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&
      (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&
      (lhs.eps() == rhs.eps()) &&
      (lhs.weight_decay() == rhs.weight_decay() &&
       (lhs.amsgrad() == rhs.amsgrad()));
}

// 将 AdamOptions 对象序列化到输出存档中
void AdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(amsgrad);
}

// 从输入存档中反序列化 AdamOptions 对象
void AdamOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, amsgrad);
}

// 获取当前学习率 lr_
double AdamOptions::get_lr() const {
  return lr();
}

// 设置当前学习率 lr_
void AdamOptions::set_lr(const double lr) {
  this->lr(lr);
}

// 判断两个 AdamParamState 对象是否相等的重载运算符
bool operator==(const AdamParamState& lhs, const AdamParamState& rhs) {
  return (lhs.step() == rhs.step()) &&
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&
      torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&
      torch::equal_if_defined(lhs.max_exp_avg_sq(), rhs.max_exp_avg_sq());
}

// 将 AdamParamState 对象序列化到输出存档中
void AdamParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_exp_avg_sq);
}

// 从输入存档中反序列化 AdamParamState 对象
void AdamParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, max_exp_avg_sq);
}

// Adam 优化器的 step 函数实现
Tensor Adam::step(LossClosure closure) {
  // 禁用梯度计算
  NoGradGuard no_grad;
  // 初始化损失张量
  Tensor loss = {};
  // 如果损失闭包不为空，开启自动梯度并计算损失
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  // 遍历每个参数组
  for (auto& group : param_groups_) {
    for (auto& p : group.params()) {
      // 遍历优化器组中的参数列表
      if (!p.grad().defined()) {
        continue;
      }
      // 获取当前参数的梯度
      auto grad = p.grad();
      // 检查梯度是否稀疏，Adam 优化器不支持稀疏梯度
      TORCH_CHECK(!grad.is_sparse(), "Adam does not support sparse gradients" /*, please consider SparseAdam instead*/);
      // 查找当前参数在优化器状态中的状态信息
      auto param_state = state_.find(p.unsafeGetTensorImpl());
      // 获取当前优化器组的配置选项
      auto& options = static_cast<AdamOptions&>(group.options());

      // 状态初始化
      if (param_state == state_.end()) {
        // 如果参数状态不存在，则创建新的参数状态
        auto state = std::make_unique<AdamParamState>();
        // 设置步数为0
        state->step(0);
        // 初始化梯度值的指数移动平均
        state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
        // 初始化梯度平方的指数移动平均
        state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
        // 如果启用了 AMSGrad，则初始化最大梯度平方指数移动平均
        if (options.amsgrad()) {
          // 维护所有二阶梯度平均值的最大值
          state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
        }
        // 将参数状态存入状态字典中
        state_[p.unsafeGetTensorImpl()] = std::move(state);
      }

      // 获取当前参数的状态引用
      auto& state =
          static_cast<AdamParamState&>(*state_[p.unsafeGetTensorImpl()]);
      // 获取当前参数状态的梯度指数移动平均和梯度平方指数移动平均
      auto& exp_avg = state.exp_avg();
      auto& exp_avg_sq = state.exp_avg_sq();
      auto& max_exp_avg_sq = state.max_exp_avg_sq();

      // 更新步数
      state.step(state.step() + 1);
      // 获取 beta1 和 beta2 参数
      auto beta1 = std::get<0>(options.betas());
      auto beta2 = std::get<1>(options.betas());

      // 计算偏置修正
      auto bias_correction1 = 1 - std::pow(beta1, state.step());
      auto bias_correction2 = 1 - std::pow(beta2, state.step());

      // 如果有权重衰减，则加入到梯度中
      if (options.weight_decay() != 0) {
        grad = grad.add(p, options.weight_decay());
      }

      // 更新梯度指数移动平均和梯度平方指数移动平均
      exp_avg.mul_(beta1).add_(grad, 1 - beta1);
      exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

      // 计算分母 denom
      Tensor denom;
      if (options.amsgrad()) {
        // 维护到目前为止所有二阶梯度平均值的最大值
        torch::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
        // 使用最大值来归一化梯度的运行平均值
        denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2))
                    .add_(options.eps());
      } else {
        denom =
            (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
      }

      // 计算步长
      auto step_size = options.lr() / bias_correction1;
      // 更新参数
      p.addcdiv_(exp_avg, denom, -step_size);
    }
  }
  // 返回损失值
  return loss;
}

void Adam::save(serialize::OutputArchive& archive) const {
  // 将当前 Adam 对象序列化保存到输出存档中
  serialize(*this, archive);
}

void Adam::load(serialize::InputArchive& archive) {
  // 声明一个 IValue 变量用于存储 PyTorch 版本信息
  IValue pytorch_version;
  // 尝试从存档中读取键为 "pytorch_version" 的值
  if (archive.try_read("pytorch_version", pytorch_version)) {
    // 若成功读取版本信息，则使用当前存档进行反序列化
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    // 如果存档使用旧的序列化格式（1.5.0 版本之前的），发出警告信息
    TORCH_WARN(
        "Your serialized Adam optimizer is still using the old serialization format. "
        "You should re-save your Adam optimizer to use the new serialization format.");
    // 声明用于存储各种缓冲区的向量
    std::vector<int64_t> step_buffers;
    std::vector<at::Tensor> exp_average_buffers;
    std::vector<at::Tensor> exp_average_sq_buffers;
    std::vector<at::Tensor> max_exp_average_sq_buffers;
    // 使用 torch::optim::serialize 函数序列化各个缓冲区到存档中
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    torch::optim::serialize(
        archive, "exp_average_buffers", exp_average_buffers);
    torch::optim::serialize(
        archive, "exp_average_sq_buffers", exp_average_sq_buffers);
    torch::optim::serialize(
        archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
    // 在 1.5.0 版本之前，假设所有张量都属于同一个 param_group
    // 获取第一个 param_group 中的所有参数张量
    std::vector<Tensor> params = param_groups_.at(0).params();
    // 遍历所有缓冲区的大小
    for (const auto idx : c10::irange(step_buffers.size())) {
      // 创建一个新的 AdamParamState 对象，并设置其各个属性
      auto state = std::make_unique<AdamParamState>();
      state->step(step_buffers.at(idx));
      state->exp_avg(exp_average_buffers.at(idx));
      state->exp_avg_sq(exp_average_sq_buffers.at(idx));
      // 如果当前索引小于 max_exp_average_sq_buffers 的大小，则设置其值
      if (idx < max_exp_average_sq_buffers.size()) {
        state->max_exp_avg_sq(max_exp_average_sq_buffers.at(idx));
      }
      // 将创建的状态对象存储到 state_ 映射中，键为参数张量的实现
      state_[params.at(idx).unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
} // namespace optim
} // namespace torch
```