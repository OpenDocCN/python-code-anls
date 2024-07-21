# `.\pytorch\aten\src\ATen\templates\RegisterBackendSelect.cpp`

```
// 使用较高优先级的调度键（BackendSelect）注册操作，优先于常规的后端特定键（如 CPU），使工厂函数调用到这里。
// 然后“手动”计算一个较低优先级以重新调度（例如 CPU），以达到最终正确的后端。
// ${generated_comment}

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS    // 定义以限制仅方法操作符
#include <ATen/core/Tensor.h>                 // 引入 ATen 核心张量库头文件
#include <ATen/core/dispatch/DispatchKeyExtractor.h>  // 引入调度键提取器头文件
#include <torch/library.h>                    // 引入 Torch 库

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Operators.h>                   // 引入 ATen 操作符头文件
#else
#include <ATen/ops/is_pinned_ops.h>           // 引入 ATen is_pinned 操作头文件
#include <ATen/ops/_pin_memory_ops.h>         // 引入 ATen _pin_memory 操作头文件

${ops_headers}                              // 插入自定义操作头文件
#endif

namespace at {

namespace {

${backend_select_method_definitions}         // 插入后端选择方法定义

bool is_pinned(const Tensor& self, std::optional<at::Device> device) {
  // 只有 CPU 张量可以被固定
  if (!self.is_cpu()) {
    return false;
  }
  // TODO: 从张量中获取标量类型？但这并不重要...
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(at::kCUDA)));
  return at::_ops::is_pinned::redispatch(_dk, self, device);
}

at::Tensor _pin_memory(const Tensor& self, std::optional<at::Device> device) {
  TORCH_CHECK(self.device().is_cpu(), "cannot pin '", self.toString(), "' only dense CPU tensors can be pinned");
  DispatchKeySet _dk = c10::DispatchKeySet(c10::computeDispatchKey(c10::nullopt, self.layout(), device.value_or(at::kCUDA)));
  if (self.is_nested()) {
    constexpr auto nested_key_set = c10::DispatchKeySet(
        {c10::DispatchKey::NestedTensor, c10::DispatchKey::AutogradNestedTensor});
    _dk = _dk.add(self.key_set() & nested_key_set);
  }
  return at::_ops::_pin_memory::redispatch(_dk, self, device);
}

// 实现 ATen 库的后端选择
TORCH_LIBRARY_IMPL(aten, BackendSelect, m) {
  ${backend_select_function_registrations};  // 注册后端选择功能函数
  m.impl(TORCH_SELECTIVE_NAME("aten::is_pinned"), TORCH_FN(is_pinned));  // 实现 is_pinned 函数的后端选择
  m.impl(TORCH_SELECTIVE_NAME("aten::_pin_memory"), TORCH_FN(_pin_memory));  // 实现 _pin_memory 函数的后端选择
}

} // namespace
} // at
```