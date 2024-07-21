# `.\pytorch\torch\csrc\autograd\saved_variable.cpp`

```py
// 包含 Torch 自动求导模块的头文件 saved_variable.h

#include <torch/csrc/autograd/saved_variable.h>

// 包含 Torch 自动求导模块的相关头文件
#include <torch/csrc/autograd/anomaly_mode.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/variable.h>

// 包含 ATen 张量的头文件
#include <ATen/Tensor.h>

// 包含 C++ 标准库的头文件
#include <memory>
#include <sstream>

// Torch 命名空间
namespace torch {
namespace autograd {

// SavedVariable 类的构造函数实现
SavedVariable::SavedVariable(
    const Variable& variable,       // 输入的变量对象
    bool is_output,                 // 是否为输出
    bool is_inplace_on_view) {      // 是否为视图上的原地操作

  // 检查变量是否已定义
  if (variable.defined()) {
    // 检查是否为推断（inference）张量，不允许保存推断张量用于反向传播
    TORCH_CHECK(
        !variable.is_inference(),
        "Inference tensors cannot be saved for backward. To work around "
        "you can make a clone to get a normal tensor and use it in autograd.")

    // 如果变量定义有效
    was_default_constructed_ = false;   // 标记未使用默认构造
    saved_version_ = variable._version();  // 记录变量版本号
    is_leaf_ = variable.is_leaf();      // 是否为叶子节点
    is_output_ = is_output;             // 是否为输出
    is_inplace_on_view_ = is_inplace_on_view;  // 是否为视图上的原地操作

    // 如果是视图上的原地操作
    if (is_inplace_on_view) {
      TORCH_INTERNAL_ASSERT(!is_leaf_ && is_output);
      weak_grad_fn_ = variable.grad_fn();  // 记录变量的梯度函数
    }

    auto maybe_hooks = get_default_hooks();  // 获取默认的钩子函数

    // 避免包装数字类型的张量泄漏给用户
    if (maybe_hooks && !variable.unsafeGetTensorImpl()->is_wrapped_number()) {
      save_metadata(variable);             // 保存元数据信息
      set_hooks_and_pack_data(std::move(maybe_hooks), variable);  // 设置钩子函数和打包数据
      return;                             // 返回
    }

    // 如果变量是叶子节点或不是输出，则可以安全保存原始变量，避免引用循环的风险
    // 1. 如果变量不是输出，其梯度函数已完全创建，特别是将与当前构造的节点（拥有此 SavedVariable 的节点）不同。
    // 2. 如果变量是叶子节点，它只对底层张量实现有弱引用，而不对梯度函数有强引用。
    // 如果变量不是输出或者是叶节点（leaf），则保存原始变量并不需要进一步处理。
    if (!is_output || is_leaf_) {
      saved_original_ = true;
      data_ = variable;
      return;
    }

    // 保存变量的元数据信息
    save_metadata(variable);

    // 只有在确实需要时才执行以下操作。
    data_ = variable.tensor_data();
  }
}

void SavedVariable::save_metadata(const Variable& data) {
    // 保存输出编号、版本计数器和需要梯度的情况下保存前向梯度

    // 获取数据的输出编号
    output_nr_ = data.output_nr();

    // 如果当前变量是叶子节点
    if (is_leaf_) {
        // 计算梯度累加器
        grad_accumulator_ = impl::grad_accumulator(data);
        // 是否需要梯度
        requires_grad_ = data.requires_grad();
    }
    // 如果不是输出节点
    else if (!is_output_) {
        // 获取梯度函数
        grad_fn_ = data.grad_fn();
    }

    // TODO(albanD) This needs to be updated when moving to multiple levels
    // 获取前向梯度
    const auto& fw_grad = data._fw_grad(/* level */ 0);
    if (fw_grad.defined()) {
        // 创建并设置前向梯度对象
        fw_grad_ = std::make_shared<ForwardGrad>();
        fw_grad_->set_value(fw_grad, /* level */ 0);
    }
}

std::unique_ptr<SavedVariableHooks> SavedVariable::get_default_hooks() {
    // 获取默认的保存变量钩子
    return Engine::get_default_engine().get_default_saved_variable_hooks();
}

void SavedVariable::reset_data() {
    // 重置数据
    hooks_.reset();
    grad_fn_.reset();
    data_.reset();
}

SavedVariable::SavedVariable(
    const std::optional<Variable>& variable,
    bool is_output,
    bool is_inplace_on_view)
    : SavedVariable(
          variable.has_value() ? *variable : Variable(),
          is_output,
          is_inplace_on_view) {}

Variable SavedVariable::unpack(std::shared_ptr<Node> saved_for) const {
    // 如果是默认构造的，则返回一个空变量
    if (was_default_constructed_) {
        return Variable();
    }

    // 如果数据未定义，则抛出错误
    if (!data_.defined()) {
        TORCH_CHECK(hooks_, ERR_BACKWARD_TWICE);
    }

    // 在版本不匹配时，希望在这里提供最有帮助的调试消息
    auto grad_fn = is_inplace_on_view_ ? weak_grad_fn_.lock()
        : !hooks_ ? saved_original_ ? data_.grad_fn() : nullptr
                  : grad_fn_;

    // 如果不是叶子节点并且梯度函数为空
    if (!is_leaf_ && !grad_fn) {
        // 当我们添加逻辑以保存原始数据时，引入了此问题
        // 现在我们依赖于 data_.grad_fn()，但如果保存的张量的自动求导元数据使用了原地分离而清除，这可能是不可靠的
        // 作为一个简单的修复，我们选择在这里禁止该行为，尽管这会根据保存的是输入还是输出而使行为不一致。
        TORCH_CHECK(
            saved_for,
            "Trying to use a saved tensor that has been detached in-place, i.e. with .detach_()."
            "This is not supported, please use out-of-place `.detach()` instead");
        grad_fn = std::move(saved_for);
    }

    // 只在没有钩子的情况下检查版本计数器
    // 如果用户提供了钩子，我们无法通过钩子跟踪版本
    if (!hooks_) {
        // 获取当前版本号
        auto current_version = impl::version_counter(data_).current_version();
    // 检查保存的版本号与当前版本号是否一致，如果不一致则执行下面的代码块
    if (saved_version_ != current_version) {
      // 创建一个字符串流对象message，用于存储错误消息
      std::stringstream message;
      // 构建错误消息内容，说明由于原地操作修改了梯度计算所需的变量之一
      message
          << "one of the variables needed for gradient computation has been "
             "modified by an inplace operation: ["
          << data_.toString() << " ";
      // 如果变量是嵌套的，则添加嵌套尺寸信息到错误消息中
      if (data_.is_nested()) {
        message << data_._nested_tensor_size() << "]";
      } else {
        message << data_.sizes() << "]";
      }
      // 如果存在梯度函数（grad_fn），则添加梯度函数输出编号和函数名称到错误消息中
      if (grad_fn) {
        message << ", which is output " << output_nr_ << " of "
                << grad_fn->name() << ",";
      }
      // 添加当前版本号和期望的版本号到错误消息中
      message << " is at version " << current_version << "; expected version "
              << saved_version_ << " instead.";
      // 根据异常模式的启用状态添加提示信息到错误消息中
      if (!AnomalyMode::is_enabled()) {
        message << " Hint: enable anomaly detection to find the operation "
                   "that failed to compute its gradient, with torch.autograd."
                   "set_detect_anomaly(True).";
      } else {
        message
            << " Hint: the backtrace further above shows the operation "
               "that failed to compute its gradient. The variable in question "
               "was changed in there or anywhere later. Good luck!";
      }
      // 抛出异常，并使用message.str()作为异常信息
      TORCH_CHECK(false, message.str());
    }
  }

  // 版本计数器是正确的。
  // 另外，如果处理非叶子变量，我们有其正确的grad_fn。

  // 如果我们有原始变量，直接返回它
  if (!hooks_ && saved_original_) {
    return data_;
  }

  // 根据是否存在hooks调用对应的unpack_hook()方法，获取数据变量data
  const auto data = hooks_ ? hooks_->call_unpack_hook() : data_;

  // 注意：保存的视图被解包为普通的变量（而不是视图），即使它们仍然共享相同的存储。
  // 这仅在我们从未对解包后的变量调用原地函数时才有效。
  Variable var;
  // 如果存在梯度函数（grad_fn），则使用make_variable创建带有Edge的变量var
  if (grad_fn) {
    var = make_variable(data, Edge(std::move(grad_fn), output_nr_));
  } else {
    // 否则，使用make_variable创建不带Edge的变量var
    var = make_variable(data, requires_grad_);
  }

  // 设置变量var的梯度累加器
  impl::set_grad_accumulator(var, grad_accumulator_);
  // 设置变量var的版本计数器，与数据变量data的版本计数器相同
  impl::set_version_counter(var, impl::version_counter(data));

  // 注意：这里的var永远不是视图，因此不需要为数据变量是视图的情况做任何特殊处理。
  // 整个论点基于此函数返回的Tensor永远不会被原地修改。
  // 如果存在前向梯度（fw_grad）且不为空，则更新它
  if (fw_grad_ && !fw_grad_->empty()) {
    // TODO(albanD) 当切换到多级时需要更新此处
    auto new_fw_grad = fw_grad_->value(/* level */ 0);
    var._set_fw_grad(new_fw_grad, /* level */ 0, /* is_inplace_op */ false);
  }

  // 返回构建好的变量var
  return var;
}

// 设置钩子函数并打包数据
void SavedVariable::set_hooks_and_pack_data(
    std::unique_ptr<SavedVariableHooks>&& hooks,
    const Variable& data) {
  // 将传入的钩子函数移动给成员变量
  hooks_ = std::move(hooks);
  // 进入无梯度区域
  at::NoGradGuard guard;
  // 获取数据的当前版本号
  const auto version = impl::version_counter(data).current_version();
  // 调用钩子函数的打包操作
  hooks_->call_pack_hook(saved_original_ ? data.detach() : data);
  // 检查数据版本号是否一致，防止钩子函数就地修改数据
  TORCH_CHECK(
      version == impl::version_counter(data).current_version(),
      "A saved tensor pack hook is modifying its input in place. "
      "Tensors provided as input to pack hook can not be modified by "
      "in-place operations as this can lead to unexpected side-effects. "
      "Please open an issue if you need to perform in-place operations on "
      "the input to a pack hook.");
}

// 注册钩子函数
void SavedVariable::register_hooks(
    std::unique_ptr<SavedVariableHooks>&& hooks) {
  // 内部断言，确保钩子函数存在
  TORCH_INTERNAL_ASSERT(hooks);
  // 检查是否已经设置过钩子函数
  TORCH_CHECK(
      !hooks_,
      "Calling register_hooks on a saved tensor whose hooks have already been set. "
      "Hint: only one pair of hooks is allowed at a time.");
  // 如果数据未定义
  if (!data_.defined()) {
    // 如果数据不是默认构造的，则抛出错误
    if (!was_default_constructed_) {
      TORCH_CHECK(
          false,
          "Calling register_hooks on a saved tensor after it has been freed. "
          "Saved intermediate values of the graph are freed when you call "
          ".backward() or autograd.grad(). Specify retain_graph=True if you "
          "need to backward through the graph a second time or if you need to "
          "access saved variables after calling backward.");
    } else {
      // 如果数据是默认构造的，则抛出错误
      TORCH_CHECK(
          false,
          "Calling register_hooks on a saved tensor with value None is forbidden");
    }
  }
  // 如果保存原始变量，则保存元数据
  if (saved_original_) {
    save_metadata(data_);
  }
  // 设置钩子函数并打包数据
  set_hooks_and_pack_data(std::move(hooks), data_);
  // 重置数据
  data_.reset();
}

// 尝试多次反向传播时的错误信息
const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time (or directly access saved "
    "tensors after they have already been freed). Saved intermediate values "
    "of the graph are freed when you call .backward() or autograd.grad(). Specify "
    "retain_graph=True if you need to backward through the graph a second time or "
    "if you need to access saved tensors after calling backward.";

} // namespace autograd
} // namespace torch
```