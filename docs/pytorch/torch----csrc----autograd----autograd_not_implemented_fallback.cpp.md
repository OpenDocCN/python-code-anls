# `.\pytorch\torch\csrc\autograd\autograd_not_implemented_fallback.cpp`

```
// 包含 Torch 的自动求导未实现回退相关头文件
#include <torch/csrc/autograd/autograd_not_implemented_fallback.h>

// 包含 C++ 标准库中的范围遍历工具
#include <c10/util/irange.h>

// 包含 ATen 的调度工具相关头文件
#include <ATen/core/TorchDispatchUtils.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>

// 包含 C10 核心库中的 Torch 调度模式线程局部存储相关头文件
#include <c10/core/impl/TorchDispatchModeTLS.h>

// 包含 Torch 的自动求导变量类型工具相关头文件
#include <torch/csrc/autograd/VariableTypeUtils.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>

// 包含 C++ 标准库中的可选值和实用工具
#include <optional>
#include <utility>
#include <vector>

// Torch 自动求导命名空间
namespace torch {
namespace autograd {

// 未命名的命名空间，用于实现私有函数或变量
namespace {

// _foreach_tensor 函数模板，用于在堆栈中枚举张量
template <typename F>
void _foreach_tensor(
    F fn,
    torch::jit::Stack* stack,
    size_t stack_start,
    size_t size) {
  // 遍历堆栈中的张量，包括张量列表中的张量
  int idx_tensor = 0;
  for (const auto idx_arg : c10::irange(size)) {
    auto& ivalue = (*stack)[stack_start + idx_arg];
    if (ivalue.isTensor()) { // 如果是张量（可选的带有值的张量）
      const auto& tensor = ivalue.toTensor();
      fn(idx_tensor, idx_arg, tensor);
      idx_tensor++;
    } else if (ivalue.isTensorList()) { // 如果是张量列表
      for (const auto& iv : ivalue.toListRef()) {
        const auto& tensor = iv.toTensor();
        fn(idx_tensor, idx_arg, tensor);
        idx_tensor++;
      }
    }
  }
}

// 自动求导回退模式，默认为警告模式
AutogradFallbackMode kAutogradFallbackMode = AutogradFallbackMode::Warn;

} // namespace

// 设置自动求导回退模式
void setAutogradFallbackMode(AutogradFallbackMode mode) {
  TORCH_CHECK(mode != AutogradFallbackMode::Error, "NYI: mode='error'");
  kAutogradFallbackMode = mode;
}

// 获取当前自动求导回退模式
AutogradFallbackMode getAutogradFallbackMode() {
  return kAutogradFallbackMode;
}

// 发出警告，指示自动求导未实现
static void warnAutogradNotImplemented(const std::string& op_name) {
  TORCH_WARN(
      op_name,
      ": an autograd kernel was not registered to the Autograd key(s) ",
      "but we are trying to backprop through it. This may lead to silently incorrect behavior. ",
      "This behavior is deprecated and will be removed in a future version of PyTorch. ",
      "If your operator is differentiable, please ensure you have registered an "
      "autograd kernel to the correct Autograd key (e.g. DispatchKey::Autograd, "
      "DispatchKey::CompositeImplicitAutograd). If your operator is not "
      "differentiable, or to squash this warning and use the previous behavior, "
      "please register torch::CppFunction::makeFallthrough() to DispatchKey::Autograd.");
}

// 表示未实现警告的结构体，继承自 Node 类
struct WarnNotImplemented : public Node {
  WarnNotImplemented(
      std::string op_name,
      size_t num_outputs,
      edge_list&& next_edges)
      : Node(std::move(next_edges)),
        op_name(std::move(op_name)),
        num_outputs(num_outputs) {}

  WarnNotImplemented(std::string op_name, size_t num_outputs)
      : op_name(std::move(op_name)), num_outputs(num_outputs) {}

  // 应用函数，返回变量列表
  variable_list apply(variable_list&& inputs) override;

  // 操作名称和输出数量
  std::string op_name;
  size_t num_outputs;
};
// 实现 WarnNotImplemented 类的 apply 方法，用于处理未实现的自动求导操作
auto WarnNotImplemented::apply(variable_list&& inputs) -> variable_list {
  // 发出警告，说明自动求导操作尚未实现
  warnAutogradNotImplemented(op_name);
  // 创建一个空的张量向量作为输出
  std::vector<at::Tensor> output(num_outputs);
  // 返回空的输出张量向量
  return output;
}

// 定义静态函数 basicAutogradNotImplementedFallbackImpl，作为自动求导未实现的基本回退实现
static void basicAutogradNotImplementedFallbackImpl(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  // 获取操作的模式和名称
  const auto& schema = op.schema();
  const auto& op_name = schema.operator_name().name;
  // 获取操作的参数和返回值的数量
  const auto num_arguments = schema.arguments().size();
  const auto num_returns = schema.returns().size();
  // 计算堆栈中操作的起始位置
  const auto stack_start = stack->size() - num_arguments;

  // 如果处于 AutogradFallbackMode::Nothing 模式，则重新分发操作并返回
  if (getAutogradFallbackMode() == AutogradFallbackMode::Nothing) {
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
    return;
  }
  // 断言当前处于 Warn 模式
  TORCH_INTERNAL_ASSERT(
      getAutogradFallbackMode() == AutogradFallbackMode::Warn);

  // 检查是否有任何输入需要梯度
  bool any_input_requires_grad = false;
  _foreach_tensor(
      [&](size_t _, size_t idx_arg, const at::Tensor& t) {
        if (t.requires_grad()) {
          any_input_requires_grad = true;
        }
      },
      stack,
      stack_start,
      num_arguments);
  
  // 优化：TLS 访问可能较慢，因此仅在需要时才进行检查
  any_input_requires_grad = any_input_requires_grad && GradMode::is_enabled();

  std::shared_ptr<WarnNotImplemented> grad_fn;

  // 如果有任何输入需要梯度，则创建 WarnNotImplemented 的共享指针
  if (any_input_requires_grad) {
    // 收集堆栈中所有张量的引用
    std::vector<const at::Tensor*> all_tensors_on_stack;
    _foreach_tensor(
        [&](size_t _, size_t idx_arg, const at::Tensor& t) {
          all_tensors_on_stack.push_back(&t);
        },
        stack,
        stack_start,
        num_arguments);
    // 创建 WarnNotImplemented 对象，并设置其 next_edges
    grad_fn = std::shared_ptr<WarnNotImplemented>(
        new WarnNotImplemented(op_name, all_tensors_on_stack.size()),
        deleteNode);
    grad_fn->set_next_edges(collect_next_edges(all_tensors_on_stack));
  }

  // 重新分发操作到下一个 Autograd Keyset
  op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);

  // 如果有任何输入需要梯度，则继续执行下面的逻辑
  if (any_input_requires_grad) {
    // 如果操作在原地对输入进行了变异并且未将其作为输出返回，则无法延迟发出警告
    // 这通常不会发生，因为大多数操作都会使用 autograd.Function 手动注册自动求导内核
    // 因此，我们期望不会出现这种情况
    // NB: 表示 "nota bene"，意为“请注意”
    _foreach_tensor(
        [&](size_t _, size_t idx_ret, const at::Tensor& t) {
          // 遍历张量，对每个张量进行操作
          if (!isDifferentiableType(t.scalar_type())) {
            // 如果张量类型不可微分，则跳过
            return;
          }
          const bool is_mutable_output =
              // 检查输出是否可变且可别名
              schema.is_aliasing({c10::SchemaArgType::output, idx_ret}) &&
              schema.is_mutable({c10::SchemaArgType::output, idx_ret});

          // 如果后向自动求导实现返回需要梯度的张量，则安装一个钩子，在反向传播时发出警告
          //
          // 注意：如果操作是原位操作且输入是视图，则可能会重新基于历史记录，钩子不会在所有应该发出警告的地方发出警告。也就是说，以下情况不会发出警告：
          // >>> x = torch.randn(3, 3, requires_grad=True)
          // >>> z = x.clone()
          // >>> w = z[0]
          // >>> k = w[0]
          // >>> y = op(k)
          // >>> torch.autograd.grad(z.sum(), w)
          if (t.requires_grad()) {
            // 注册一个钩子函数，用于在反向传播时发出警告
            t.register_hook([op_name](const at::Tensor& grad) {
              warnAutogradNotImplemented(op_name);
            });
            // 如果历史记录被重新基于，则尝试在视图的基础上发出警告。这将捕获大多数情况（因为用户通常调用.backward()并通过整个程序进行反向传播）。
            if (t.is_view() && is_mutable_output) {
              const auto& base = t._base();
              if (base.requires_grad()) {
                // 只能在需要梯度的张量上注册钩子
                base.register_hook([op_name](const at::TensorBase& grad) {
                  warnAutogradNotImplemented(op_name);
                });
              }
            }
            return;
          }

          // 如果后向自动求导实现返回任何不需要梯度的张量，则安装 WarnNotImplemented grad_fn。这个 grad_fn 在反向传播时发出警告并返回未定义的张量梯度。
          //
          // 注意 [自动求导回退和原位操作]
          // 如果模式表明输出是可变的，并且输出是一个输入，并且输入是一个视图张量，则...
          // 我们不确定是否可以执行 set_history，所以我们跳过添加 grad_fn。内置操作符在这里重新基于历史记录，但自定义操作符可能有多个张量(a!)返回值，rebase_history 假设单个张量(a!)返回值，在一般情况下，自定义操作符没有一个良好的原位操作故事。
          if (!is_mutable_output) {
            // 设置张量的历史记录
            set_history(t, grad_fn);
          }
        },
        stack,
        stack->size() - num_returns,
        num_returns);
  }
}

// 定义一个名为 basicAutogradNotImplementedFallback 的函数，返回类型为 torch::CppFunction
torch::CppFunction basicAutogradNotImplementedFallback() {
  // 返回一个从 boxed 函数中创建的 CppFunction，具体函数实现为 basicAutogradNotImplementedFallbackImpl
  return torch::CppFunction::makeFromBoxedFunction<&basicAutogradNotImplementedFallbackImpl>();
}

// 定义 VariableHooks 类的成员函数 basic_autograd_not_implemented_fallback
void VariableHooks::basic_autograd_not_implemented_fallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) const {
  // 调用 basicAutogradNotImplementedFallbackImpl 函数来实现基本的自动求导未实现回退逻辑
  basicAutogradNotImplementedFallbackImpl(op, dispatch_keys, stack);
}

// 定义静态函数 autogradNotImplementedFallbackImpl
static void autogradNotImplementedFallbackImpl(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  // 模仿 VariableType 的未实现内核的逻辑子集
  // 详见 gen_variable_type.py
  const auto& schema = op.schema();
  const auto& op_name = schema.operator_name().name;
  const auto num_arguments = schema.arguments().size();
  const auto num_returns = schema.returns().size();
  const auto stack_start = stack->size() - num_arguments;
  const bool grad_mode = GradMode::is_enabled();
  std::vector<const at::Tensor*> tensors_requiring_grad_on_stack;

  // 跟踪哪些输出是就地修改的输出，以便必要时重新设置历史记录
  std::vector<bool> is_inplace_output(num_returns, false);
  bool any_is_inplace_output = false;
  std::vector<bool> is_aliased_output(num_returns, false);
  std::optional<size_t> aliased_output_idx;

  // 遍历每个返回值，检查是否有就地修改的输出或者别名输出
  for (const auto i : c10::irange(num_returns)) {
    if (schema.is_aliasing({c10::SchemaArgType::output, i})) {
      if (schema.is_mutable({c10::SchemaArgType::output, i})) {
        is_inplace_output[i] = true;
        any_is_inplace_output = true;
      } else {
        TORCH_CHECK(
            !aliased_output_idx.has_value(),
            "Expected only a single output in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
            "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
            "Please rewrite your function as a composite function.");
        aliased_output_idx = i;
      }
      is_aliased_output[i] = true;
    }
  }

  int64_t aliased_input_idx = -1;
  // 遍历每个输入参数，检查是否有输入参数是别名输入，但不可变
  for (const auto i : c10::irange(num_arguments)) {
    if (schema.is_aliasing({c10::SchemaArgType::input, i}) &&
        !schema.is_mutable({c10::SchemaArgType::input, i})) {
      TORCH_CHECK(
          aliased_input_idx == -1,
          "Expected only a single input in the operator schema to have a non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple inputs are aliased with outputs aren't supported. "
          "Please rewrite your function as a composite function.");
      aliased_input_idx = static_cast<int64_t>(i);
  }
}

size_t num_tensor_inputs = 0; // 仅用于调试检查

// 对于堆栈中的每个张量执行操作
_foreach_tensor(
    [&](size_t _, size_t idx_arg, const at::Tensor& t) {
      // 如果处于梯度计算模式并且张量需要梯度，则将其添加到需要梯度的张量列表中
      if (grad_mode && t.requires_grad()) {
        tensors_requiring_grad_on_stack.push_back(&t);
      }
      // 增加张量输入计数
      num_tensor_inputs++;
      // 检查是否支持使用前向自动微分
      TORCH_CHECK_NOT_IMPLEMENTED(
          !isFwGradDefined(t),
          "Trying to use forward AD with ",
          op_name,
          " that does not support it.");
    },
    stack,
    stack_start,
    num_arguments);

// 检查是否存在任何需要梯度的张量
const bool any_requires_grad = !tensors_requiring_grad_on_stack.empty();
// 检查函数是否具有输出参数
const bool has_out_arg = std::any_of(
    schema.arguments().begin(),
    schema.arguments().end(),
    [](const c10::Argument& arg) { return arg.is_out(); });

// 对于堆栈中的每个张量执行操作
_foreach_tensor(
    [&](size_t _, size_t i, const at::Tensor& t) {
      // 如果模式可变，并且参数是可变的
      if (schema.is_mutable({c10::SchemaArgType::input, i})) {
        // 如果存在输出参数
        if (has_out_arg) {
          // 通常情况下，输出参数重载不支持任何需要梯度的参数。
          // 但是，为了保持向后兼容性，我们放宽了这个检查。
          // 参见 https://github.com/pytorch/pytorch/issues/120988
          if (can_mutate_inplace(t, any_requires_grad) !=
              can_mutate_inplace_result::success) {
            // 如果不能就地修改，则抛出错误
            throw_error_out_requires_grad(schema.name().c_str());
          }
        } else {
          // 检查是否可以就地修改
          check_inplace(t, any_requires_grad);
        }
      }
    },
    stack,
    stack_start,
    num_arguments);

// 创建一个未实现的梯度函数对象，如果有任何需要梯度的张量
std::shared_ptr<NotImplemented> grad_fn;
if (any_requires_grad) {
  grad_fn = std::shared_ptr<NotImplemented>(
      new NotImplemented(op_name), deleteNode);
  // 设置梯度函数对象的下一个边缘
  grad_fn->set_next_edges(
      collect_next_edges(tensors_requiring_grad_on_stack));
}
#ifndef NDEBUG
  // 如果未定义 NDEBUG 宏，则执行以下代码块（仅在调试模式下有效）
  // NOTE [ TensorImpl and Storage Pointer Sanity Checks ]
  // 注意：用于对 TensorImpl 和 Storage 指针进行健全性检查

  // 创建 stack_args_copy，复制 stack 中指定范围的元素到新的 vector
  auto stack_args_copy =
      std::vector<c10::IValue>(stack->begin() + stack_start, stack->end());

  // 创建 impl_saved，用于保存 TensorImpl 的智能指针
  std::vector<c10::intrusive_ptr<c10::TensorImpl>> impl_saved;
  impl_saved.reserve(num_tensor_inputs);

  // 创建 storage_saved，用于保存 Storage 的 optional 对象
  std::vector<std::optional<c10::Storage>> storage_saved;
  storage_saved.reserve(num_tensor_inputs);

  // 对每个 Tensor 执行以下操作：
  _foreach_tensor(
      [&](size_t idx, size_t _, const at::Tensor& t) {
        // 如果 Tensor 有有效的 Storage，则将其存储在 storage_saved 中
        storage_saved.push_back(
            t.has_storage() ? std::optional<c10::Storage>(t.storage())
                            : c10::nullopt);
        // 将 Tensor 的 IntrusivePtr 存储在 impl_saved 中
        impl_saved.push_back(t.getIntrusivePtr());
      },
      &stack_args_copy,
      0,
      num_arguments);
#endif

// 如果存在别名输入索引或任何一个操作是 inplace 输出
if (aliased_input_idx != -1 || any_is_inplace_output) {
  // 创建一个 AutoDispatchBelowAutograd 保护块
  at::AutoDispatchBelowAutograd guard;
  // 使用 op 对象重新分派操作，根据 dispatch_keys 和 c10::after_autograd_keyset
  op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
} else {
  // 如果既不是 inplace 也不是 view 操作
  // 创建一个 AutoDispatchBelowADInplaceOrView 保护块
  at::AutoDispatchBelowADInplaceOrView guard;
  // 使用 op 对象重新分派操作，根据 dispatch_keys 和 c10::after_ADInplaceOrView_keyset
  op.redispatchBoxed(
      dispatch_keys & c10::after_ADInplaceOrView_keyset, stack);
}
#ifndef NDEBUG
  // Iterate over each tensor in the stack and perform assertions if debug mode is enabled
  _foreach_tensor(
      [&](size_t idx_tensor, size_t _, const at::Tensor& t) {
        // Check if the storage of the current tensor matches the saved storage
        if (storage_saved.at(idx_tensor).has_value())
          TORCH_INTERNAL_ASSERT(
              storage_saved.at(idx_tensor).value().is_alias_of(t.storage()),
              op_name);
        // Check if the implementation pointer of the current tensor matches the saved pointer
        if (impl_saved.at(idx_tensor))
          TORCH_INTERNAL_ASSERT(
              impl_saved.at(idx_tensor) == t.getIntrusivePtr(), op_name);
      },
      &stack_args_copy, // Reference to the stack arguments
      0,                // Starting index in the stack
      num_arguments);   // Number of tensors to iterate over

  // Iterate over each tensor in the stack that is to be returned and perform assertions
  _foreach_tensor(
      [&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
        // Skip certain checks based on conditions related to dispatching and operation name
        if (at::impl::tensor_has_dispatch(t) ||
            at::impl::dispatch_mode_enabled() ||
            op_name.rfind("aten::_nested_get", 0) == 0)
          return;
        // If the output tensor is not inplace, ensure it has a low reference count
        if (!is_inplace_output[idx_ret])
          TORCH_INTERNAL_ASSERT(
              t.use_count() <= 1, op_name); // It's okay for undefined tensors to be returned
        // Additional assertion related to tensor norm calculation and storage usage
        // Referencing an issue in PyTorch's GitHub repository
        if (!is_aliased_output[idx_ret] && t.has_storage() &&
            op_name != "aten::_foreach_norm")
          TORCH_INTERNAL_ASSERT(t.storage().use_count() == 1);
      },
      stack,                              // Stack of return values
      stack->size() - num_returns,        // Starting index in the stack for return values
      num_returns);                       // Number of tensors to iterate over in return values

  // Ensure that there is exactly one pair of aliased input-output tensors
  if (aliased_input_idx != -1 && aliased_output_idx.has_value()) {
    // Retrieve the aliased input and output tensors from the stack copies
    const c10::IValue& aliased_input_iv = stack_args_copy[aliased_input_idx];
    const c10::IValue& aliased_output_iv =
        (*stack)[stack->size() - num_returns + *aliased_output_idx];
    // Perform type assertions for the aliased input and output tensors
    TORCH_INTERNAL_ASSERT(aliased_input_iv.isTensor(), op_name);
    TORCH_INTERNAL_ASSERT(
        aliased_output_iv.isTensor() || aliased_output_iv.isTensorList(),
        op_name);
    // Extract the tensors from their IValue wrappers
    const at::Tensor& aliased_input = aliased_input_iv.toTensor();
    // 检查输入是否具有存储，并进一步处理
    if (aliased_input.has_storage()) {
      // 检查输出是否为张量类型
      if (aliased_output_iv.isTensor()) {
        // 获取输出张量的引用
        const at::Tensor& aliased_output = aliased_input_iv.toTensor();
        // 目前跳过对子类的断言检查
        // TODO: 修复涉及子类的别名情况
        // 检查是否禁用了调度模式，并且输入和输出张量都没有调度
        if (!at::impl::dispatch_mode_enabled() &&
            !at::impl::tensor_has_dispatch(aliased_input) &&
            !at::impl::tensor_has_dispatch(aliased_output)) {
          // 断言输入存储是否与输出存储别名一致，使用操作名称作为错误信息
          TORCH_INTERNAL_ASSERT(
              aliased_input.storage().is_alias_of(aliased_output.storage()),
              op_name);
        }
      } else {
        // 获取输出张量向量
        const auto aliased_output_vec = aliased_output_iv.toTensorVector();
        // 遍历输出张量向量
        for (const auto& aliased_output : aliased_output_vec) {
          // 目前跳过对子类的断言检查
          // TODO: 修复涉及子类的别名情况
          // 检查是否禁用了调度模式，并且输入和输出张量都没有调度
          if (!at::impl::dispatch_mode_enabled() &&
              !at::impl::tensor_has_dispatch(aliased_input) &&
              !at::impl::tensor_has_dispatch(aliased_output)) {
            // 断言输入存储是否与输出存储别名一致，使用操作名称作为错误信息
            TORCH_INTERNAL_ASSERT(
                aliased_input.storage().is_alias_of(aliased_output.storage()),
                op_name);
          }
        }
      }
    }
  }
#endif

  // 如果任何张量需要梯度
  if (any_requires_grad) {
    // 遍历堆栈中的张量
    _foreach_tensor(
        [&](size_t idx_tensor, size_t idx_ret, const at::Tensor& t) {
          // 检查张量是否可微分
          if (isDifferentiableType(t.scalar_type())) {
            // 如果是就地操作的输出张量
            if (is_inplace_output[idx_ret]) {
              // 将张量的历史重新基于给定的梯度函数
              rebase_history(t, grad_fn);
            } else {
              // 设置张量的历史为给定的梯度函数
              set_history(t, grad_fn);
            }
          }
        },
        stack,
        stack->size() - num_returns,
        num_returns);
  }
}

// 返回一个CppFunction对象，其实现调用autogradNotImplementedFallbackImpl函数
torch::CppFunction autogradNotImplementedFallback() {
  return torch::CppFunction::makeFromBoxedFunction<
      &autogradNotImplementedFallbackImpl>();
}

// 实现处理不支持原位或视图操作的回退逻辑
static void autogradNotImplementedInplaceOrViewFallbackImpl(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  // 模拟ADInplaceOrViewType内核的部分逻辑
  // 参见gen_inplace_or_view_type.py
  // 仅与autogradNotImplementedFallback一起使用
  // 更多信息请参考https://pytorch.org/tutorials/advanced/dispatcher

  // 注意：ADInplaceOrView boxed kernel的限制
  // 此操作应仅与autogradNotImplementedFallback内核一起使用
  // 因为我们需要一些逻辑来确保即使在此内核中创建的视图上进行原地操作，仍会正确地引发“导数未实现”错误。

  const auto& schema = op.schema();
  const auto& op_name = schema.operator_name().name;
  const auto num_arguments = schema.arguments().size();
  const auto num_returns = schema.returns().size();
  const auto stack_start = stack->size() - num_arguments;

  at::Tensor aliased_input;

  int64_t aliased_output_idx = -1;
  for (const auto i : c10::irange(num_returns)) {
    // 检查输出是否具有别名，且不可变
    if (schema.is_aliasing({c10::SchemaArgType::output, i}) &&
        !schema.is_mutable({c10::SchemaArgType::output, i})) {
      // 检查只有一个输出在操作模式中具有非写入别名注释
      TORCH_CHECK(
          aliased_output_idx == -1,
          "Fallback ADInplaceOrView kernel expects only a single output in the operator schema to have a "
          "non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple outputs are aliased with inputs aren't supported."
          "Please rewrite your function as a composite function.");
      aliased_output_idx = static_cast<int64_t>(i);
    }
  }

  std::optional<size_t> aliased_input_idx;
  for (const auto i : c10::irange(num_arguments)) {
    // 检查输入是否具有别名
    // 检查当前输入是否存在别名，并且这个输入不可变
    if (schema.is_aliasing({c10::SchemaArgType::input, i}) &&
        !schema.is_mutable({c10::SchemaArgType::input, i})) {
      // 断言确保只有一个输入被标记为非写入别名（如 'Tensor(a)'）
      TORCH_CHECK(
          !aliased_input_idx.has_value(),
          "Fallback ADInplaceOrView kernel expects only a single input in the operator schema to have a "
          "non-write alias annotation (i.e., 'Tensor(a)'). "
          "Non-composite functions where multiple inputs are aliased with outputs aren't supported. "
          "Please rewrite your function as a composite function.");
      // 记录第一个被标记为别名的输入的索引
      aliased_input_idx = i;
      // 获取栈上索引为 stack_start + i 的 IValue 引用
      const c10::IValue& aliased_input_iv =
          (*stack)[stack_start + i]; // get a reference to an ivalue on the
                                     // stack
      // 断言确保获取的别名输入是 Tensor 类型
      TORCH_CHECK(aliased_input_iv.isTensor());
      // 将别名输入转换为 Tensor，并赋给 aliased_input
      aliased_input =
          aliased_input_iv.toTensor(); // TODO: Can we avoid saving this tensor
                                       // and incurring the refcount bump?
    }
  }
  // 检查是否存在视图关系，即是否有别名输入
  // 查看上面的注释 [ Limitations of ADInplaceOrView boxed kernel ]
  TORCH_CHECK(
      (!aliased_input_idx.has_value() && aliased_output_idx == -1) ||
          (aliased_input_idx.has_value() && aliased_input_idx.value() == 0 &&
           aliased_output_idx == 0),
      "Fallback ADInplaceOrView kernel can only create view relationships between the first "
      "input and the first output (the output can be a vector of tensors). Please change the "
      "order of your operator's parameters so that this is the case.");
  // 判断是否存在视图关系
  const bool is_view = aliased_input_idx.has_value();

  {
    // 进入临时的 ADInplaceOrView 以下自动分发的作用域
    at::AutoDispatchBelowADInplaceOrView guard;
    // 重新分发操作，使用之前定义的 dispatch_keys 和 stack
    op.redispatchBoxed(
        dispatch_keys & c10::after_ADInplaceOrView_keyset, stack);
  }

  // 遍历所有的返回值
  for (const auto i : c10::irange(num_returns)) {
    // 检查当前输出是否可变
    if (schema.is_mutable({c10::SchemaArgType::output, i})) {
      // 增加当前输出的版本号
      increment_version((*stack)[stack->size() - num_returns + i].toTensor());
    }
  }

  // 如果存在视图关系
  if (is_view) {
    // 获取栈上索引为 stack->size() - num_returns + aliased_output_idx 的 IValue 引用
    c10::IValue& aliased_output_iv =
        (*stack)[stack->size() - num_returns + aliased_output_idx];

    // 查看上面的注释 [ View + Inplace detection ] 了解更多关于这段逻辑的详情
    // 创建一个错误消息，指示不能对没有实现导数的视图进行修改
    auto error_msg =
        ("Mutating the view " + op_name +
         "which does not have a derivative implemented is forbidden.");
    // 创建一个错误视图函数，用于抛出错误
    auto erroring_view_func = std::make_unique<ErroringViewFunc>(error_msg);

    // 创建一个错误的逆视图函数，用于抛出错误
    const auto erroring_rev_view_func = [op_name = op_name](const at::Tensor&) {
      TORCH_CHECK(
          false,
          "Accessing the reverse view for ",
          op_name,
          " which does not have a derivative implemented is forbidden.");
      return at::Tensor();
    };
    // 检查 aliased_output_iv 是否为 TensorList
    if (aliased_output_iv.isTensorList()) {
      // 将 aliased_output_iv 转换为 Tensor 向量
      auto aliased_output = aliased_output_iv.toTensorVector();
      // 遍历 aliased_output 中的每个子 Tensor
      for (auto& sub_output : aliased_output) {
        // 调用 as_view 函数来创建视图
        as_view(
            /* base=*/aliased_input,  // 视图的基础张量
            /* tensor=*/sub_output,   // 视图所针对的子张量
            /* is_bw_differentiable=*/true,  // 是否支持反向传播的不同iable
            /* is_fw_differentiable=*/true,  // 是否支持前向传播的不同iable
            /* view_func=*/std::move(erroring_view_func),  // 错误视图函数
            /* rev_view_func=*/erroring_rev_view_func,     // 反向视图函数
            /* creation_meta=*/
            InferenceMode::is_enabled()
                ? CreationMeta::INFERENCE_MODE  // 创建模式为推理模式
                : (at::GradMode::is_enabled() ? CreationMeta::MULTI_OUTPUT_NODE  // 创建模式为多输出节点
                                              : CreationMeta::NO_GRAD_MODE));  // 创建模式为无梯度模式
      }
      // 将 aliased_output 移动到 result 中
      auto result = std::move(aliased_output);
      // 将 result 存储到 stack 中特定位置
      stack->at(stack->size() - num_returns + aliased_output_idx) = result;
    } else {
      // 若 aliased_output_iv 不是 TensorList，则需满足其为 Tensor
      TORCH_CHECK(aliased_output_iv.isTensor());
      // 将 aliased_output_iv 转换为 Tensor，并创建视图
      auto result = as_view(
          /* base=*/aliased_input,  // 视图的基础张量
          /* tensor=*/std::move(aliased_output_iv).toTensor(),  // 视图所针对的张量
          /* is_bw_differentiable=*/true,  // 是否支持反向传播的不同iable
          /* is_fw_differentiable=*/true,  // 是否支持前向传播的不同iable
          /* view_func=*/std::move(erroring_view_func),  // 错误视图函数
          /* rev_view_func=*/erroring_rev_view_func,     // 反向视图函数
          /* creation_meta=*/
          InferenceMode::is_enabled()
              ? CreationMeta::INFERENCE_MODE  // 创建模式为推理模式
              : (at::GradMode::is_enabled() ? CreationMeta::DEFAULT  // 创建模式为默认
                                            : CreationMeta::NO_GRAD_MODE));  // 创建模式为无梯度模式
      // 将 result 移动到 stack 中特定位置
      stack->at(stack->size() - num_returns + aliased_output_idx) =
          std::move(result);
    }
  }
}

// 定义一个名为 autogradNotImplementedInplaceOrViewFallback 的函数，返回一个 torch::CppFunction 对象
torch::CppFunction autogradNotImplementedInplaceOrViewFallback() {
    // 使用 autogradNotImplementedInplaceOrViewFallbackImpl 函数创建一个 torch::CppFunction 对象并返回
    return torch::CppFunction::makeFromBoxedFunction<&autogradNotImplementedInplaceOrViewFallbackImpl>();
}

} // 结束 autograd 命名空间
} // 结束 torch 命名空间
```