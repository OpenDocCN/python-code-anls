# `.\pytorch\torch\csrc\api\src\optim\adagrad.cpp`

```py
/// 包含 torch 库中 Adagrad 优化器的头文件
#include <torch/optim/adagrad.h>

/// 包含 torch 的变量自动求导和序列化相关的头文件
#include <torch/csrc/autograd/variable.h>
#include <torch/optim/serialize.h>
#include <torch/serialize/archive.h>
#include <torch/utils.h>

/// 包含 ATen 库中的 Tensor 操作头文件
#include <ATen/ATen.h>
#include <c10/util/irange.h>

/// 包含 C++ 标准库中的函数对象头文件
#include <functional>

/// 定义 torch 命名空间下的 optim 命名空间
namespace torch {
namespace optim {

/// AdagradOptions 类的构造函数，初始化学习率 lr_
AdagradOptions::AdagradOptions(double lr) : lr_(lr) {}

/// 比较两个 AdagradOptions 对象是否相等的重载运算符
bool operator==(const AdagradOptions& lhs, const AdagradOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.lr_decay() == rhs.lr_decay()) &&
      (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.initial_accumulator_value() == rhs.initial_accumulator_value()) &&
      (lhs.eps() == rhs.eps());
}

/// 将 AdagradOptions 对象序列化为输出存档
void AdagradOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(initial_accumulator_value);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);
}

/// 从输入存档中反序列化 AdagradOptions 对象
void AdagradOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, initial_accumulator_value);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);
}

/// 获取学习率 lr_ 的值
double AdagradOptions::get_lr() const {
  return lr();
}

/// 设置学习率 lr_ 的值
void AdagradOptions::set_lr(const double lr) {
  this->lr(lr);
}

/// 比较两个 AdagradParamState 对象是否相等的重载运算符
bool operator==(const AdagradParamState& lhs, const AdagradParamState& rhs) {
  return (lhs.step() == rhs.step()) && torch::equal(lhs.sum(), rhs.sum());
}

/// 将 AdagradParamState 对象序列化为输出存档
void AdagradParamState::serialize(
    torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(sum);
}

/// 从输入存档中反序列化 AdagradParamState 对象
void AdagradParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, sum);
}

/// Adagrad 类的实现，参考自 https://github.com/pytorch/pytorch/blob/master/torch/optim/adagrad.py
/// 实现了 step 方法，用于更新模型参数
Tensor Adagrad::step(LossClosure closure) {
  NoGradGuard no_grad; /// 禁用梯度计算的上下文管理器
  Tensor loss = {}; /// 初始化 loss Tensor
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true); /// 开启自动求导模式
    loss = closure(); /// 执行损失函数计算，并获取损失值
  }
  for (auto& group : param_groups_) { /// 遍历优化器的参数组
    // 遍历优化器参数组中的每个参数 p
    for (auto& p : group.params()) {
      // 如果参数的梯度未定义，跳过该参数
      if (!p.grad().defined()) {
        continue;
      }
      // 获取参数的梯度
      auto grad = p.grad();
      // 断言确保状态中存在该张量对应的状态信息
      TORCH_INTERNAL_ASSERT(
          state_[p.unsafeGetTensorImpl()] != nullptr,
          "state found NULL for the Tensor ",
          p);
      // 将状态信息转换为 AdagradParamState 类型的引用
      auto& state =
          static_cast<AdagradParamState&>(*state_[p.unsafeGetTensorImpl()]);
      // 获取优化器参数组的选项信息
      auto& options = static_cast<AdagradOptions&>(group.options());

      // 更新步数计数器
      state.step(state.step() + 1);

      // 如果设置了权重衰减
      if (options.weight_decay() != 0) {
        // 检查梯度是否是稀疏的，稀疏梯度与权重衰减选项不兼容
        TORCH_CHECK(
            !p.grad().is_sparse(),
            "weight_decay option is not compatible with sparse gradients");
        // 将梯度加上权重衰减项
        grad = grad.add(p, options.weight_decay());
      }
      // 计算当前步长率 clr
      const auto clr = options.lr() /
          (1 + static_cast<double>(state.step() - 1) * options.lr_decay());

      // 如果梯度是稀疏的
      if (grad.is_sparse()) {
        // 将梯度稀疏化
        grad = grad.coalesce();
        auto grad_indices = grad._indices();
        auto grad_values = grad._values();
        auto size = grad.sizes();

        // 定义一个函数，用于生成稀疏张量
        auto make_sparse = [&](const Tensor& values) -> Tensor {
          if (grad_indices.dim() == 0 || values.dim() == 0) {
            return torch::empty({0}, grad.options()).resize_as_(grad);
          }
          return torch::sparse_coo_tensor(
              grad_indices, values, size, grad.options());
        };
        // 更新状态信息中的 sum，加上梯度值的平方
        state.sum(state.sum().add_(make_sparse(grad_values.pow(2))));
        // 对状态信息进行稀疏掩码操作，得到标准差
        auto std = state.sum().sparse_mask(grad);
        // 计算标准差的值，并加上一个小的常数 eps，用于数值稳定性
        const auto std_values = std._values().sqrt_().add_(options.eps());

        // 更新参数 p 的值，使用稀疏梯度和标准差进行更新
        p.add_(make_sparse(grad_values / std_values), -clr);
      } else {
        // 更新状态信息中的 sum，使用梯度的平方和
        state.sum(state.sum().addcmul_(grad, grad, 1.0));
        // 计算标准差，并加上一个小的常数 eps，用于数值稳定性
        const auto std = state.sum().sqrt().add_(options.eps());
        // 更新参数 p 的值，使用梯度和标准差进行更新
        p.addcdiv_(grad, std, -clr);
      }
    }
  }
  // 返回计算得到的损失值
  return loss;
}

void Adagrad::save(serialize::OutputArchive& archive) const {
  serialize(*this, archive);
}


# Adagrad 类的 save 方法，用于将对象序列化保存到存档中
void Adagrad::save(serialize::OutputArchive& archive) const {
  // 使用自定义的序列化函数将当前对象 (*this) 序列化到输出存档中
  serialize(*this, archive);
}



void Adagrad::load(serialize::InputArchive& archive) {
  // 定义一个 IValue 对象，用于存储读取的 pytorch_version 信息
  IValue pytorch_version;
  // 尝试从存档中读取 "pytorch_version"，如果成功读取到，则说明是新格式的存档
  if (archive.try_read("pytorch_version", pytorch_version)) {
    // 使用自定义的序列化函数将存档中的内容反序列化到当前对象 (*this) 中
    serialize(*this, archive);
  } else { // deserializing archives saved in old format (prior to
           // version 1.5.0)
    // 如果无法读取到 "pytorch_version"，则说明是旧格式的存档（1.5.0 版本之前）
    TORCH_WARN(
        "Your serialized Adagrad optimizer is still using the old serialization format. "
        "You should re-save your Adagrad optimizer to use the new serialization format.");
    // 定义两个空的 Tensor 容器，用于存储 sum_buffers 和 step_buffers
    std::vector<Tensor> sum_buffers;
    std::vector<int64_t> step_buffers;
    // 使用 torch::optim::serialize 函数从存档中读取 "sum_buffers" 和 "step_buffers" 的内容
    torch::optim::serialize(archive, "sum_buffers", sum_buffers);
    torch::optim::serialize(archive, "step_buffers", step_buffers);
    // 在版本 1.5.0 之前，不存在 param_groups，所有的张量都假设在一个 param_group 中
    // 获取第一个 param_group 的所有参数
    std::vector<Tensor> params = param_groups_.at(0).params();
    // 遍历参数列表
    for (const auto idx : c10::irange(params.size())) {
      // 创建一个 AdagradParamState 对象的智能指针，用于存储参数的状态信息
      auto state = std::make_unique<AdagradParamState>();
      // 设置 state 的 step 属性为 step_buffers[idx] 的值
      state->step(step_buffers[idx]);
      // 设置 state 的 sum 属性为 sum_buffers[idx] 的值
      state->sum(sum_buffers[idx]);
      // 将当前参数 params[idx] 的不安全张量实现与 state 的智能指针关联，并存储到 state_ 中
      state_[params[idx].unsafeGetTensorImpl()] = std::move(state);
    }
  }
}


} // namespace optim
} // namespace torch


# 命名空间闭合：结束 optim 命名空间和 torch 命名空间的定义
} // namespace optim
} // namespace torch
```