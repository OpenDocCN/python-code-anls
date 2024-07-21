# `.\pytorch\aten\src\ATen\core\MetaFallbackKernel.cpp`

```py
// 包含头文件，定义了 ATen 核心的调度器 Dispatcher 和 PyInterpreter 类
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/impl/PyInterpreter.h>
// 包含了 torch 库的头文件，用于注册和定义 Torch 库的内容
#include <torch/library.h>

// 命名空间 at::impl，实现了一些内部函数和类
namespace at::impl {

// 定义了静态函数 metaFallback，处理当操作使用 Meta 张量时的回退情况
static void metaFallback(
    const c10::OperatorHandle& op,        // 操作符句柄，表示正在处理的操作
    c10::DispatchKeySet dispatch_keys,    // 分发键集合，表示允许的分发键
    torch::jit::Stack* stack) {           // Torch JIT 的栈指针，用于操作的参数堆栈管理
  // 检查操作是否包含 Python 模块，如果包含则抛出异常
  c10::Dispatcher::singleton().throwIfHasPythonModule(op.operator_name());
  // 抛出未实现异常，提示无法在 Meta 张量上运行操作的错误信息
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      op.operator_name(),
      ": attempted to run this operator with Meta tensors, but there was no ",
      "fake impl or Meta kernel registered. You may have run into this message "
      "while using an operator with PT2 compilation APIs (torch.compile/torch.export); "
      "in order to use this operator with those APIs you'll need to add a fake impl. "
      "Please see the following for next steps:  "
      "https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html");
}

// 在 Meta 库中注册一个回退函数，使其与 Torch 库进行绑定
TORCH_LIBRARY_IMPL(_, Meta, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&metaFallback>());
}

} // namespace at::impl
```