# `.\pytorch\aten\src\ATen\core\dispatch\backend_fallback_test.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Functions.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <c10/util/irange.h>
#include <torch/library.h>

using namespace at;

namespace {

// This test file gives an example of a simple use case for "wrapper"
// and "mode" style tensor type ids.  In both cases, the implementation
// of the wrapper/mode simply passes through the call to underlying JIT
// implementation (so the wrapper/mode doesn't actually do anything),
// but this could be used as a starting point to do more interesting things.

// Global counter for ease of testing
static int64_t override_call_count = 0;

// Mode implementation

// 处理模式的回退函数，调用操作符并增加计数
void generic_mode_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  override_call_count++;
  // 禁用指定的分发键（DispatchKey::TESTING_ONLY_GenericMode）
  c10::impl::ExcludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericMode);
  op.callBoxed(stack);
}

// Wrapper implementation

// 通用包装器张量实现，继承自TensorImpl
struct GenericWrapperTensorImpl : public c10::TensorImpl {
  explicit GenericWrapperTensorImpl(at::Tensor rep)
    : TensorImpl(
        c10::DispatchKeySet(c10::DispatchKey::TESTING_ONLY_GenericWrapper),
        rep.dtype(),
        rep.device()
        // TODO: propagate size!
      )
    , rep_(std::move(rep)) {}

  at::Tensor rep_;
};

// 处理包装器的回退函数，调用操作符并增加计数
void generic_wrapper_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  override_call_count++;

  auto num_arguments = op.schema().arguments().size();
  auto num_returns = op.schema().returns().size();

  // 解包所有参数
  auto args = torch::jit::pop(*stack, num_arguments);
  for (const auto i : c10::irange(num_arguments)) {
    // TODO: Handle tensor list
    if (args[i].isTensor()) {
      auto* impl = args[i].unsafeToTensorImpl();
      // 检查是否是包装器张量实现
      if (impl->key_set().has(DispatchKey::TESTING_ONLY_GenericWrapper)) {
        auto* wrapper = static_cast<GenericWrapperTensorImpl*>(impl);
        torch::jit::push(*stack, wrapper->rep_);  // 不移动！
      } else {
        torch::jit::push(*stack, std::move(args[i]));
      }
    } else {
      torch::jit::push(*stack, std::move(args[i]));
    }
  }

  // 调用操作符
  op.callBoxed(stack);

  // 重新包装输出
  auto rets = torch::jit::pop(*stack, num_returns);
  for (const auto i : c10::irange(num_returns)) {
    // TODO: Handle tensor list
    if (rets[i].isTensor()) {
      // 创建GenericWrapperTensorImpl类型的张量实现并推送到栈上
      torch::jit::push(*stack, at::detail::make_tensor<GenericWrapperTensorImpl>(std::move(rets[i]).toTensor()));  // 移动！
    } else {
      torch::jit::push(*stack, std::move(rets[i]));
    }
  }
}

#ifndef ATEN_CPU_STATIC_DISPATCH
// 测试用例：BackendFallbackTest，测试后端回退功能（带模式）
TEST(BackendFallbackTest, TestBackendFallbackWithMode) {
  // 创建一个 Torch 库的实现对象 m，使用 TESTING_ONLY_GenericMode 模式
  auto m = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericMode);
  // 将 generic_mode_fallback 函数设置为回退函数
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&generic_mode_fallback>());

  // 设置 DispatchKey 为 TESTING_ONLY_GenericMode 的包含分发键保护
  c10::impl::IncludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericMode);

  // 重置 override_call_count 计数为 0
  override_call_count = 0;
  // 创建一个大小为 5x5，元素为 1.0 的双精度张量 a
  Tensor a = ones({5, 5}, kDouble);
  // 对张量 a 进行批归一化操作
  Tensor b = batch_norm(a, {}, {}, {}, {}, true, 0.1, 1e-05, false);
  // 断言 override_call_count 应为 2
  ASSERT_EQ(override_call_count, 2);
}

// 测试用例：BackendFallbackTest，测试后端回退功能（使用包装器）
TEST(BackendFallbackTest, TestBackendFallbackWithWrapper) {
  // 创建一个 Torch 库的实现对象 m，使用 TESTING_ONLY_GenericWrapper 模式
  auto m = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericWrapper);
  // 将 generic_wrapper_fallback 函数设置为回退函数
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&generic_wrapper_fallback>());

  // 重置 override_call_count 计数为 0
  override_call_count = 0;
  // 创建一个 GenericWrapperTensorImpl 类型的张量 a，大小为 5x5，元素为 1.0 的双精度张量
  Tensor a = at::detail::make_tensor<GenericWrapperTensorImpl>(ones({5, 5}, kDouble));
  // 对张量 a 进行批归一化操作
  Tensor b = batch_norm(a, {}, {}, {}, {}, true, 0.1, 1e-05, false);
  // 断言 override_call_count 应为 1
  ASSERT_EQ(override_call_count, 1);
}

// 测试用例：BackendFallbackTest，测试后端回退功能（通过 fallthrough）
TEST(BackendFallbackTest, TestFallthroughBackendFallback) {
  // 创建一个 aten 库的实现对象 m，使用 TESTING_ONLY_GenericMode 模式
  auto m = MAKE_TORCH_LIBRARY_IMPL(aten, TESTING_ONLY_GenericMode);
  // 将 mul.Tensor 的实现设置为 generic_mode_fallback 函数
  m.impl("mul.Tensor", torch::CppFunction::makeFromBoxedFunction<&generic_mode_fallback>());

  // 创建一个 Torch 库的实现对象 gm，使用 TESTING_ONLY_GenericMode 模式
  auto gm = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericMode);
  // 将回退函数设置为 fallthrough（继续向下执行）
  gm.fallback(torch::CppFunction::makeFallthrough());

  // 设置 DispatchKey 为 TESTING_ONLY_GenericMode 的包含分发键保护
  c10::impl::IncludeDispatchKeyGuard guard(DispatchKey::TESTING_ONLY_GenericMode);

  // 重置 override_call_count 计数为 0
  override_call_count = 0;
  // 创建一个大小为 5x5，元素为 0.0 的双精度张量 a
  // 不会触发回退，因为我们使用 fallthrough
  Tensor a = zeros({5, 5}, kDouble);
  // 断言 override_call_count 应为 0
  ASSERT_EQ(override_call_count, 0);
  // 创建一个大小为 5x5，元素为 a 的双精度张量 b
  // 会触发回退，因为我们显式设置了它
  Tensor b = mul(a, a);
  // 断言 override_call_count 应为 1
  ASSERT_EQ(override_call_count, 1);
}
#endif // ATEN_CPU_STATIC_DISPATCH
```