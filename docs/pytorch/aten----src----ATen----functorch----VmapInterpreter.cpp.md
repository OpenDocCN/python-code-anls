# `.\pytorch\aten\src\ATen\functorch\VmapInterpreter.cpp`

```
#include <ATen/functorch/VmapInterpreter.h>
#include <ATen/functorch/DynamicLayer.h>

namespace at::functorch {

void VmapInterpreterPtr::processImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack) {
  // 设置当前线程的分发键为 Vmap，指定 DispatchKeySet 为 FuncTorchVmapMode
  setup_dispatch_key_tls(TransformType::Vmap, DispatchKeySet(DispatchKey::FuncTorchVmapMode));
  // 调用操作符的 boxed 调用方法，传入栈 stack
  op.callBoxed(stack);
}

void VmapInterpreterPtr::sendToNextInterpreterImpl(
    const c10::OperatorHandle& op,
    torch::jit::Stack* stack,
    bool grad_special_case) {
  // 如果动态层栈为空，则进行栈的完整性检查
  if (getDynamicLayerStack().empty()) {
    sanityCheckStack(op, stack);
  }
  // 调用操作符的 boxed 调用方法，传入栈 stack
  op.callBoxed(stack);
}

} // namespace at::functorch
```