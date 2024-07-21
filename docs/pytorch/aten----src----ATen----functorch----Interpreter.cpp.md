# `.\pytorch\aten\src\ATen\functorch\Interpreter.cpp`

```py
// 包含所需的头文件以实现Functorch的Interpreter
#include <ATen/functorch/Interpreter.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/TensorWrapper.h>
#include <ATen/functorch/VmapInterpreter.h>
#include <ATen/functorch/FunctionalizeInterpreter.h>
#include <ATen/functorch/ADInterpreters.h>
#include <ATen/functorch/DynamicLayer.h>

// 定义了functorch命名空间
namespace at::functorch {

// 返回所有动态层调度键的集合
static DispatchKeySet get_all_dynlayer_keyset() {
  // 注意：FULL_AFTER不包括DispatchKey
  // "包括DynamicLayer{Front, Back}Mode之间的所有调度键"
  auto result =
    DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::FuncTorchDynamicLayerFrontMode) -
    DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::FuncTorchDynamicLayerBackMode);
  // 添加DispatchKey::FuncTorchDynamicLayerFrontMode
  result = result | DispatchKeySet({DispatchKey::FuncTorchDynamicLayerFrontMode});

  // 禁用自动转换调度键的处理，因为它们与functorch的交互很奇怪
  result = result - autocast_dispatch_keyset;

  // 禁用DispatchKey::FuncTorchVmapMode的处理，需要更好的建模方式
  // 例如，在grad(vmap(f))中，DispatchKey::FuncTorchVmapMode导致所有随机操作，
  // 即使在处理vmap层完成后，仍然会报错。
  result = result.remove(DispatchKey::FuncTorchVmapMode);

  return result;
}

// 所有动态层调度键的静态全局变量
static DispatchKeySet all_dynlayer_keyset = get_all_dynlayer_keyset();

// 返回进入动态层时要包含的调度键集合，根据给定的TransformType
static DispatchKeySet keysForEnteringDynamicLayer(TransformType key) {
  if (key == TransformType::Vmap) {
    // 注意：不包括DispatchKey::FuncTorchVmapMode。在构造DynamicLayer时可能会修改调度键，
    // 但在进入/退出DynamicLayer时不控制它。
    return DispatchKeySet({DispatchKey::FuncTorchBatched, DispatchKey::BatchedNestedTensor});
  } else if (key == TransformType::Grad || key == TransformType::Jvp) {
    return autograd_dispatch_keyset.add(DispatchKey::ADInplaceOrView);
  } else if (key == TransformType::Functionalize) {
    return DispatchKeySet(DispatchKey::Functionalize);
  } else {
    TORCH_INTERNAL_ASSERT(false, "Unsupported key: ", key);
  }
}

// 返回进入动态层时要排除的调度键集合，根据给定的TransformType
DispatchKeySet keysToExcludeWhenEnteringDynamicLayer(TransformType key) {
  DispatchKeySet exclude = all_dynlayer_keyset;
  exclude = exclude.remove(DispatchKey::FuncTorchDynamicLayerBackMode);
  exclude = exclude - keysForEnteringDynamicLayer(key);
  return exclude;
}

// 设置当前线程局部存储中的调度键，根据给定的TransformType和额外要包含的调度键集合
void setup_dispatch_key_tls(TransformType key, DispatchKeySet also_include) {
  auto local_keyset = c10::impl::tls_local_dispatch_key_set();
  auto to_exclude = local_keyset.excluded_;
  to_exclude = to_exclude | keysToExcludeWhenEnteringDynamicLayer(key);
  to_exclude = to_exclude - keysForEnteringDynamicLayer(key);
  local_keyset.excluded_ = to_exclude;
  local_keyset.included_ = local_keyset.included_ | also_include;
  c10::impl::_force_tls_local_dispatch_key_set(local_keyset);
}

} // namespace at::functorch
std::ostream& operator<<(std::ostream& os, const TransformType& t) {
  // 根据枚举类型 TransformType 输出相应的字符串表示到输出流 os
  switch (t) {
    case TransformType::Torch:
      os << "Torch";
      break;
    case TransformType::Vmap:
      os << "Vmap";
      break;
    case TransformType::Grad:
      os << "Grad";
      break;
    case TransformType::Jvp:
      os << "Jvp";
      break;
    case TransformType::Functionalize:
      os << "Functionalize";
      break;
    default:
      // 如果遇到未知的 TransformType，触发内部断言错误
      TORCH_INTERNAL_ASSERT(false);
  }
  return os;
}

void sanityCheckStack(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 获取操作符 op 的参数个数
  auto num_args = op.schema().arguments().size();
  // 对于栈中从 stack->size() - num_args 到 stack->size() 的每个张量，进行检查
  foreachTensorInplace(*stack, static_cast<int64_t>(stack->size() - num_args), static_cast<int64_t>(stack->size()),
      [](const Tensor& tensor) {
        // 解包死亡引用的张量，并获取张量的包装器
        auto result = unwrapIfDead(tensor);
        auto* wrapper = maybeGetTensorWrapper(result);
        // 断言：张量的包装器应为 nullptr
        TORCH_INTERNAL_ASSERT(wrapper == nullptr);
        // 尝试获取批处理实现，应为 nullptr
        auto* batched = maybeGetBatchedImpl(result);
        TORCH_INTERNAL_ASSERT(batched == nullptr);
        // 返回原始张量
        return tensor;
      });
}

#define INTERPRETER_DISPATCH(type, method) \
  switch (key()) { \
    // 根据枚举类型 key() 分发到不同的处理器实例，并调用相应的 method
    case TransformType::Vmap: \
      TORCH_INTERNAL_ASSERT(std::holds_alternative<VmapInterpreterMeta>(this->meta()));\
      return VmapInterpreterPtr(this). method; \
    case TransformType::Grad: \
      TORCH_INTERNAL_ASSERT(std::holds_alternative<GradInterpreterMeta>(this->meta()));\
      return GradInterpreterPtr(this). method; \
    case TransformType::Jvp: \
      TORCH_INTERNAL_ASSERT(std::holds_alternative<JvpInterpreterMeta>(this->meta()));\
      return JvpInterpreterPtr(this). method; \
    case TransformType::Functionalize: \
      TORCH_INTERNAL_ASSERT(std::holds_alternative<FunctionalizeInterpreterMeta>(this->meta()));\
      return FunctionalizeInterpreterPtr(this). method; \
    default: \
      // 如果遇到未知的 TransformType，触发内部断言错误，并输出错误信息 "Unrecognized transform"
      TORCH_INTERNAL_ASSERT(false, "Unrecognized transform"); \
  }

void Interpreter::process(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  // 使用 INTERPRETER_DISPATCH 宏根据 key_ 调度到对应的处理函数 processImpl
  INTERPRETER_DISPATCH(key_, SINGLE_ARG(processImpl(op, stack)));
}

void Interpreter::sendToNextInterpreter(const c10::OperatorHandle& op, torch::jit::Stack* stack, bool grad_special_case) {
  // 使用 INTERPRETER_DISPATCH 宏根据 key_ 调度到对应的处理函数 sendToNextInterpreterImpl
  INTERPRETER_DISPATCH(key_, SINGLE_ARG(sendToNextInterpreterImpl(op, stack, grad_special_case)));
}

} // namespace at::functorch
```