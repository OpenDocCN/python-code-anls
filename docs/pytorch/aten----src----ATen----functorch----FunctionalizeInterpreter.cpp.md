# `.\pytorch\aten\src\ATen\functorch\FunctionalizeInterpreter.cpp`

```py
namespace at::functorch {
```  

static void sanityCheckNotFunctional(const c10::OperatorHandle& op, torch::jit::Stack* stack, size_t num_args) {
  // 对于输入栈中的每个张量，检查是否存在 FunctionalTensor，如果存在则抛出异常
  foreachTensorInplace(*stack, stack->size() - num_args, stack->size(),
      [](const Tensor& tensor) {
        TORCH_INTERNAL_ASSERT(!at::functionalization::impl::isFunctionalTensor(tensor));
        return tensor;
      });
}
```py  

void FunctionalizeInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  // 如果在层栈中开启了 functionalize()，则始终调用 functionalization 内核
  // functionalization 内核负责根据输入张量是否为 functional，选择是否执行操作
  setup_dispatch_key_tls(TransformType::Functionalize, DispatchKeySet(DispatchKey::Functionalize));
  auto functionalization_add_back_views = functionalizeAddBackViews();
  // 设置一个 TLS（线程局部存储），用于切换 functionalization 的行为
  // 如果设置，则 functionalization 仅移除变异操作，而不移除视图操作
  at::functionalization::impl::FunctionalizationReapplyViewsGuard functional_guard(functionalization_add_back_views);

  // 调用操作的 boxed 版本
  op.callBoxed(stack);

  // 检查返回值的张量，设置其 functional wrapper 的级别
  auto ret_size = op.schema().returns().size();
  foreachTensorInplace(*stack, stack->size() - ret_size, stack->size(),
    [&](const Tensor& tensor) {
      if (at::functionalization::impl::isFunctionalTensor(tensor)) {
        auto wrapper = at::functionalization::impl::unsafeGetFunctionalWrapper(tensor);
        // 设置 wrapper 的级别，由 functorch 负责设置，因为核心中没有此信息
        // 对于工厂操作，我们无法在 functionalize 内核中简单地“传播”级别
        wrapper->set_level(level());
      }
      return tensor;
    }
  );
}
```  

void FunctionalizeInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool grad_special_case) {
  // 暂时不支持嵌套的 functionalization 调用，确保在后退模式回退时，解开 FunctionalTensor
  auto args_size = op.schema().arguments().size();
  sanityCheckNotFunctional(op, stack, args_size);

  // 重新分发操作
  if (getDynamicLayerStack().empty()) {
    sanityCheckStack(op, stack);
  }
  op.callBoxed(stack);

  // 检查返回值的张量，确保没有 FunctionalTensor
  auto ret_size = op.schema().returns().size();
  sanityCheckNotFunctional(op, stack, ret_size);
}
```py  

} // namespace at::functorch
```