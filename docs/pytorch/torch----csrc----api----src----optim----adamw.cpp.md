# `.\pytorch\torch\csrc\api\src\optim\adamw.cpp`

```
#include <torch/optim/adamw.h>  // 引入AdamW优化器的头文件

#include <torch/csrc/autograd/variable.h>  // 引入变量自动求导相关头文件
#include <torch/nn/module.h>  // 引入神经网络模块相关头文件
#include <torch/serialize/archive.h>  // 引入序列化存档相关头文件
#include <torch/utils.h>  // 引入工具函数相关头文件

#include <ATen/ATen.h>  // 引入ATen张量库相关头文件
#include <c10/util/irange.h>  // 引入C10实用库中的irange头文件

#include <cmath>  // 引入数学库中的数学函数
#include <functional>  // 引入函数式编程相关功能

namespace torch {
namespace optim {

AdamWOptions::AdamWOptions(double lr) : lr_(lr) {}  // AdamW选项构造函数，初始化学习率

bool operator==(const AdamWOptions& lhs, const AdamWOptions& rhs) {
  return (lhs.lr() == rhs.lr()) &&  // 比较学习率是否相等
      (std::get<0>(lhs.betas()) == std::get<0>(rhs.betas())) &&  // 比较beta1是否相等
      (std::get<1>(lhs.betas()) == std::get<1>(rhs.betas())) &&  // 比较beta2是否相等
      (lhs.eps() == rhs.eps()) &&  // 比较epsilon是否相等
      (lhs.weight_decay() == rhs.weight_decay()) &&  // 比较权重衰减是否相等
      (lhs.amsgrad() == rhs.amsgrad());  // 比较AMSGrad是否开启
}

void AdamWOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);  // 序列化学习率
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(betas);  // 序列化beta值
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(eps);  // 序列化epsilon
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);  // 序列化权重衰减
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(amsgrad);  // 序列化AMSGrad标志
}

void AdamWOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);  // 反序列化学习率
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(betas_t, betas);  // 反序列化beta值
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, eps);  // 反序列化epsilon
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);  // 反序列化权重衰减
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, amsgrad);  // 反序列化AMSGrad标志
}

double AdamWOptions::get_lr() const {
  return lr();  // 返回当前学习率
}

void AdamWOptions::set_lr(const double lr) {
  this->lr(lr);  // 设置新的学习率
}

bool operator==(const AdamWParamState& lhs, const AdamWParamState& rhs) {
  return (lhs.step() == rhs.step()) &&  // 比较步数是否相等
      torch::equal(lhs.exp_avg(), rhs.exp_avg()) &&  // 比较exp_avg是否相等
      torch::equal(lhs.exp_avg_sq(), rhs.exp_avg_sq()) &&  // 比较exp_avg_sq是否相等
      torch::equal_if_defined(lhs.max_exp_avg_sq(), rhs.max_exp_avg_sq());  // 比较max_exp_avg_sq是否相等
}

void AdamWParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(step);  // 序列化步数
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg);  // 序列化exp_avg
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(exp_avg_sq);  // 序列化exp_avg_sq
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(max_exp_avg_sq);  // 序列化max_exp_avg_sq
}

void AdamWParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(int64_t, step);  // 反序列化步数
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg);  // 反序列化exp_avg
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, exp_avg_sq);  // 反序列化exp_avg_sq
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, max_exp_avg_sq);  // 反序列化max_exp_avg_sq
}

Tensor AdamW::step(LossClosure closure) {
  NoGradGuard no_grad;  // 禁用梯度自动计算
  Tensor loss = {};  // 初始化损失张量
  if (closure != nullptr) {  // 如果损失函数闭包不为空
    at::AutoGradMode enable_grad(true);  // 启用梯度自动计算模式
    loss = closure();  // 计算损失
  }
  for (auto& group : param_groups_) {  // 遍历参数组
    for (auto& p : group.params()) {
        // 遍历优化器参数组中的每个参数 p
        if (!p.grad().defined()) {
            // 如果当前参数 p 的梯度未定义，则跳过该参数
            continue;
        }
        // 获取当前参数 p 的梯度
        const auto& grad = p.grad();
        // 检查当前梯度是否为稀疏张量，AdamW 不支持稀疏梯度
        TORCH_CHECK(!grad.is_sparse(), "AdamW does not support sparse gradients" /*, please consider SparseAdamW instead*/);

        // 查找当前参数 p 对应的状态信息
        auto param_state = state_.find(p.unsafeGetTensorImpl());
        // 获取当前优化器参数组的选项
        auto& options = static_cast<AdamWOptions&>(group.options());

        // 执行权重衰减（weight decay）
        if (options.weight_decay() != 0) {
            // 更新参数 p，进行权重衰减
            p.mul_(1 - options.lr() * options.weight_decay());
        }

        // 初始化状态信息
        if (param_state == state_.end()) {
            // 如果参数状态不存在，则创建新的状态信息
            auto state = std::make_unique<AdamWParamState>();
            // 初始化步数为 0
            state->step(0);
            // 初始化梯度的指数移动平均值为零张量
            state->exp_avg(torch::zeros_like(p, MemoryFormat::Preserve));
            // 初始化梯度平方的指数移动平均值为零张量
            state->exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
            // 如果启用了 AMSGrad，则初始化最大梯度平方移动平均值为零张量
            if (options.amsgrad()) {
                // 维护所有梯度平方移动平均值的最大值
                state->max_exp_avg_sq(torch::zeros_like(p, MemoryFormat::Preserve));
            }
            // 将新创建的状态信息与参数 p 关联并保存
            state_[p.unsafeGetTensorImpl()] = std::move(state);
        }

        // 获取参数 p 对应的状态信息
        auto& state = static_cast<AdamWParamState&>(*state_[p.unsafeGetTensorImpl()]);
        // 获取当前参数 p 的梯度指数移动平均值
        auto& exp_avg = state.exp_avg();
        // 获取当前参数 p 的梯度平方指数移动平均值
        auto& exp_avg_sq = state.exp_avg_sq();
        // 获取当前参数 p 的最大梯度平方指数移动平均值（如果使用 AMSGrad）
        auto& max_exp_avg_sq = state.max_exp_avg_sq();

        // 更新步数
        state.step(state.step() + 1);
        // 获取 beta1 和 beta2 参数
        auto beta1 = std::get<0>(options.betas());
        auto beta2 = std::get<1>(options.betas());

        // 计算偏置修正项
        auto bias_correction1 = 1 - std::pow(beta1, state.step());
        auto bias_correction2 = 1 - std::pow(beta2, state.step());

        // 更新梯度的指数移动平均值和梯度平方的指数移动平均值
        exp_avg.mul_(beta1).add_(grad, 1 - beta1);
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

        // 初始化 denom 张量
        Tensor denom;
        if (options.amsgrad()) {
            // 如果启用了 AMSGrad
            // 更新最大梯度平方的指数移动平均值
            torch::max_out(max_exp_avg_sq, exp_avg_sq, max_exp_avg_sq);
            // 使用最大梯度平方的指数移动平均值来归一化梯度的运行平均值
            denom = (max_exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
        } else {
            // 如果未启用 AMSGrad
            // 使用梯度平方的指数移动平均值来归一化梯度的运行平均值
            denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(options.eps());
        }

        // 计算步长
        auto step_size = options.lr() / bias_correction1;
        // 更新参数 p
        p.addcdiv_(exp_avg, denom, -step_size);
    }
  }
  return loss;
// 实现 AdamW 类的保存函数，将当前对象序列化到输出存档中
void AdamW::save(serialize::OutputArchive& archive) const {
    // 调用全局的序列化函数，将当前对象保存到存档中
    serialize(*this, archive);
}

// 实现 AdamW 类的加载函数，从输入存档中反序列化对象数据
void AdamW::load(serialize::InputArchive& archive) {
    // 声明一个 IValue 变量用于存储序列化存档中的 pytorch_version
    IValue pytorch_version;
    // 尝试从存档中读取 pytorch_version，并存储到 pytorch_version 变量中
    if (archive.try_read("pytorch_version", pytorch_version)) {
        // 如果成功读取 pytorch_version，则调用全局的序列化函数，反序列化当前对象
        serialize(*this, archive);
    } else { // 处理旧格式的存档（1.5.0 版本之前的）
        // 发出警告，说明当前的 AdamW 优化器使用了旧的序列化格式
        TORCH_WARN(
            "Your serialized AdamW optimizer is still using the old serialization format. "
            "You should re-save your AdamW optimizer to use the new serialization format.");
        
        // 声明几个存储步骤和缓冲区的向量
        std::vector<int64_t> step_buffers;
        std::vector<at::Tensor> exp_average_buffers;
        std::vector<at::Tensor> exp_average_sq_buffers;
        std::vector<at::Tensor> max_exp_average_sq_buffers;
        
        // 使用 torch::optim 命名空间的函数，从存档中读取各个缓冲区的数据
        torch::optim::serialize(archive, "step_buffers", step_buffers);
        torch::optim::serialize(
            archive, "exp_average_buffers", exp_average_buffers);
        torch::optim::serialize(
            archive, "exp_average_sq_buffers", exp_average_sq_buffers);
        torch::optim::serialize(
            archive, "max_exp_average_sq_buffers", max_exp_average_sq_buffers);
        
        // 由于 1.5.0 版本之前没有 param_groups，假设所有张量都在同一个 param_group 中
        std::vector<Tensor> params = param_groups_.at(0).params();
        
        // 遍历步骤缓冲区的大小范围
        for (const auto idx : c10::irange(step_buffers.size())) {
            // 创建一个新的 AdamWParamState 对象，用于存储当前索引的状态信息
            auto state = std::make_unique<AdamWParamState>();
            state->step(step_buffers.at(idx));  // 设置步骤
            state->exp_avg(exp_average_buffers.at(idx));  // 设置 exp_avg
            state->exp_avg_sq(exp_average_sq_buffers.at(idx));  // 设置 exp_avg_sq
            // 如果当前索引小于 max_exp_average_sq_buffers 的大小，设置 max_exp_avg_sq
            if (idx < max_exp_average_sq_buffers.size()) {
                state->max_exp_avg_sq(max_exp_average_sq_buffers.at(idx));
            }
            // 将当前张量的状态信息存储到 state_ 中
            state_[params.at(idx).unsafeGetTensorImpl()] = std::move(state);
        }
    }
}
```