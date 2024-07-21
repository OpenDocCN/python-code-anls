# `.\pytorch\aten\src\ATen\core\boxing\KernelFunction.cpp`

```
// 引入 ATen 核心库中的头文件和调度器头文件
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>

// 引入 C++ 标准库中的字符串流
#include <sstream>

// 进入 c10 命名空间
namespace c10 {

// 这是一个“虚假”的内核函数，实际上并不执行任何操作。它是一个特殊的内核函数，
// 在调度表中被特别处理。它的语义是重新调度到下一个本应处理的调度键，跳过当前的调度键。
void fallthrough_kernel(OperatorKernel*, const OperatorHandle&, DispatchKeySet, Stack*) {
  // 内部断言，如果执行到这里，应该由调度器提前截断
  TORCH_INTERNAL_ASSERT(0,
    "fallthrough_kernel was executed but it should have been short-circuited by the dispatcher. "
    "This could occur if you registered a fallthrough kernel as a override for a specific operator "
    "(as opposed to a backend fallback); this is NOT currently supported, and we do not intend to "
    "add support for it in the near future.  If you do find yourself in need of this, "
    "let us know in the bug tracker.");
}

// 处理模糊的 AutogradOther 内核函数
void ambiguous_autogradother_kernel(OperatorKernel*, const OperatorHandle& op, DispatchKeySet, Stack*) {
  // 内部断言，说明注册了既属于 CompositeImplicitAutograd 又属于 AutogradOther 的内核函数
  TORCH_INTERNAL_ASSERT(0,
    op.operator_name(), " has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther. "
    "This makes the backend kernel unreachable; the dispatcher will always prefer the CompositeImplicitAutograd lowering "
    "(see Note [Ambiguity in AutogradOther kernel]). "
    "If you want to override CompositeImplicitAutograd, please open an issue to request a dedicated "
    "Autograd dispatch key for the backend.\n",
    "If you only want to run inference instead of training, in C++, add `c10::InferenceMode mode;` "
    "before model.forward(); in Python, use `torch.inference_mode()` as a context manager (see "
    "https://pytorch.org/docs/stable/generated/torch.inference_mode.html).",
    "\nCanonical state\n~~~~~~~~~~~\n", op.dumpState(), "\n\n");
}

// 处理不支持命名张量的内核函数
void named_not_supported_kernel(OperatorKernel*, const OperatorHandle& op, DispatchKeySet, Stack*) {
  // 检查点，说明不支持使用命名张量进行操作
  TORCH_CHECK(0,
    op.operator_name(), " is not yet supported with named tensors. Please drop names via "
    "`tensor = tensor.rename(None)`, call the op with an unnamed tensor, "
    "and set names on the result of the operation."
    );
}

// 返回内核函数状态的单行摘要
std::string KernelFunction::dumpState() const {
  // 创建字符串流对象
  std::ostringstream oss;
  // 获取封装内核函数的函数指针
  auto boxed_kernel_fn = boxed_kernel_func_.getFnPtr();
  // 如果封装内核函数指针指向 fallthrough_kernel，则追加字符串 "fallthrough "
  if (boxed_kernel_fn == fallthrough_kernel) {
    oss << "fallthrough ";
  }
  // 如果存在封装内核函数指针，则追加字符串 "boxed "
  if (boxed_kernel_fn) {
    oss << "boxed ";
  }
  // 如果存在非封装内核函数，则追加字符串 "unboxed "
  if (unboxed_kernel_func_) {
    oss << "unboxed ";
  }
  // 返回构建的状态字符串
  return oss.str();
}

// 比较封装和非封装内核函数是否相等的私有函数
bool KernelFunction::_equalsBoxedAndUnboxed(const KernelFunction& other) const {
  // 比较两个对象的封装内核函数指针和非封装内核函数指针是否相等
  return boxed_kernel_func_.getFnPtr() == other.boxed_kernel_func_.getFnPtr() &&
         unboxed_kernel_func_ == other.unboxed_kernel_func_;
}

} // namespace c10
```