# `.\pytorch\torch\csrc\api\src\optim\sgd.cpp`

```
// 包含 Torch SGD 优化器的头文件
#include <torch/optim/sgd.h>

// 包含 Torch 的变量自动求导和优化器相关头文件
#include <torch/csrc/autograd/variable.h>
#include <torch/optim/optimizer.h>
#include <torch/optim/serialize.h>
#include <torch/types.h>
#include <torch/utils.h>

// 包含 ATen 张量操作库和 C++ 便利函数库的头文件
#include <ATen/ATen.h>
#include <c10/util/irange.h>

// 包含 C++ 标准库中的函数对象
#include <functional>

// Torch 命名空间
namespace torch {
namespace optim {

// SGDOptions 类的构造函数，接受学习率作为参数
SGDOptions::SGDOptions(double lr) : lr_(lr) {}

// 判断两个 SGDOptions 对象是否相等的运算符重载
bool operator==(const SGDOptions& lhs, const SGDOptions& rhs) {
  return (lhs.lr() == rhs.lr()) && (lhs.momentum() == rhs.momentum()) &&
      (lhs.dampening() == rhs.dampening()) &&
      (lhs.weight_decay() == rhs.weight_decay()) &&
      (lhs.nesterov() == rhs.nesterov());
}

// 将 SGDOptions 对象序列化到输出存档中
void SGDOptions::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(lr);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(dampening);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(weight_decay);
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(nesterov);
}

// 从输入存档中反序列化 SGDOptions 对象
void SGDOptions::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, lr);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, momentum);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, dampening);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(double, weight_decay);
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(bool, nesterov);
}

// 获取学习率 lr 的值
double SGDOptions::get_lr() const {
  return lr();
}

// 设置学习率 lr 的值
void SGDOptions::set_lr(const double lr) {
  this->lr(lr);
}

// 判断两个 SGDParamState 对象是否相等的运算符重载
bool operator==(const SGDParamState& lhs, const SGDParamState& rhs) {
  return torch::equal(lhs.momentum_buffer(), rhs.momentum_buffer());
}

// 将 SGDParamState 对象序列化到输出存档中
void SGDParamState::serialize(torch::serialize::OutputArchive& archive) const {
  _TORCH_OPTIM_SERIALIZE_TORCH_ARG(momentum_buffer);
}

// 从输入存档中反序列化 SGDParamState 对象
void SGDParamState::serialize(torch::serialize::InputArchive& archive) {
  _TORCH_OPTIM_DESERIALIZE_TORCH_ARG(Tensor, momentum_buffer);
}

// SGD 类的 step 方法实现，执行一步优化
Tensor SGD::step(LossClosure closure) {
  // 禁用梯度计算
  NoGradGuard no_grad;
  // 初始化损失张量为空
  Tensor loss = {};
  // 如果有损失函数闭包，则开启梯度计算并计算损失
  if (closure != nullptr) {
    at::AutoGradMode enable_grad(true);
    loss = closure();
  }
  // 遍历所有参数组
  for (auto& group : param_groups_) {
    // 获取当前参数组的 SGDOptions 对象引用
    auto& options = static_cast<SGDOptions&>(group.options());
    // 获取当前参数组的权重衰减值
    auto weight_decay = options.weight_decay();
    // 获取当前参数组的动量值
    auto momentum = options.momentum();
    // 获取当前参数组的阻尼值
    auto dampening = options.dampening();
    // 获取当前参数组的 Nesterov 标志
    auto nesterov = options.nesterov();


这段代码是 Torch 中 SGD 优化器的实现部分，注释详细解释了每一行代码的功能和作用。
    // 遍历参数组 `group` 中的每个参数 `p`
    for (auto& p : group.params()) {
        // 如果当前参数的梯度未定义，则跳过该参数
        if (!p.grad().defined()) {
            continue;
        }
        // 获取当前参数 `p` 的梯度数据 `d_p`
        auto d_p = p.grad().data();
        
        // 如果有权重衰减 (weight_decay)，则对梯度数据进行权重衰减操作
        if (weight_decay != 0) {
            d_p = d_p.add(p.data(), weight_decay);
        }
        
        // 如果有动量 (momentum)，则进行动量相关操作
        if (momentum != 0) {
            // 定义缓冲张量 `buf`
            Tensor buf;
            // 查找当前参数 `p` 在状态字典 `state_` 中的状态
            auto param_state = state_.find(p.unsafeGetTensorImpl());
            // 如果找不到对应状态
            if (param_state == state_.end()) {
                // 创建动量缓冲张量 `buf`，并保存状态到 `state_` 中
                buf = torch::clone(d_p).detach();
                auto state = std::make_unique<SGDParamState>();
                state->momentum_buffer(buf);
                state_[p.unsafeGetTensorImpl()] = std::move(state);
            } else {
                // 否则，从状态中获取动量缓冲张量 `buf`
                buf = static_cast<SGDParamState&>(*param_state->second)
                          .momentum_buffer();
                // 根据动量和阻尼系数更新缓冲张量 `buf`
                buf.mul_(momentum).add_(d_p, 1 - dampening);
            }
            // 如果使用 Nesterov 动量，则更新参数梯度 `d_p`
            if (nesterov) {
                d_p = d_p.add(buf, momentum);
            } else {
                // 否则直接使用缓冲张量 `buf`
                d_p = buf;
            }
        }
        
        // 更新参数 `p` 的数据，使用学习率 `options.lr()` 的负值乘以梯度 `d_p`
        p.data().add_(d_p, -1 * options.lr());
    }
  }
  // 返回计算得到的损失值 `loss`
  return loss;
}

// 保存SGD对象的状态到输出存档中
void SGD::save(serialize::OutputArchive& archive) const {
  // 使用serialize函数将当前SGD对象序列化到给定的输出存档中
  serialize(*this, archive);
}

// 从输入存档中加载SGD对象的状态
void SGD::load(serialize::InputArchive& archive) {
  // 创建一个IValue对象来存储读取的PyTorch版本信息
  IValue pytorch_version;
  // 尝试从存档中读取键为"pytorch_version"的值，如果存在则继续反序列化
  if (archive.try_read("pytorch_version", pytorch_version)) {
    // 使用serialize函数将存档中的数据反序列化到当前SGD对象中
    serialize(*this, archive);
  } else { // 在旧格式（1.5.0版本之前）的存档中进行反序列化
    // 发出警告，指示当前SGD优化器的序列化格式仍然使用旧格式
    TORCH_WARN(
        "Your serialized SGD optimizer is still using the old serialization format. "
        "You should re-save your SGD optimizer to use the new serialization format.");
    
    // 创建一个向量来存储动量缓存Tensor
    std::vector<Tensor> momentum_buffers;
    // 从存档中反序列化"momentum_buffers"键对应的数据到向量中
    torch::optim::serialize(archive, "momentum_buffers", momentum_buffers);
    
    // 由于1.5.0版本之前没有param_groups，假设现在所有的Tensor都在一个param_group中
    std::vector<Tensor> params = param_groups_.at(0).params();
    
    // 遍历动量缓存向量的索引范围
    for (const auto idx : c10::irange(momentum_buffers.size())) {
      // 创建一个SGDParamState的独占指针state，用于管理动量缓存
      auto state = std::make_unique<SGDParamState>();
      // 将当前索引处的动量缓存Tensor设置为state的动量缓存
      state->momentum_buffer(momentum_buffers[idx]);
      // 使用Tensor的底层实现指针作为键，将state移动到状态映射state_中
      state_[params[idx].unsafeGetTensorImpl()] = std::move(state);
    }
  }
}
} // namespace optim
} // namespace torch
```