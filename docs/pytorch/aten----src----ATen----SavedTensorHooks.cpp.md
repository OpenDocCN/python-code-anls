# `.\pytorch\aten\src\ATen\SavedTensorHooks.cpp`

```py
// 引入头文件 SavedTensorHooks.h
#include <ATen/SavedTensorHooks.h>
// 引入异常处理工具 Exception.h
#include <c10/util/Exception.h>
// 引入标准库 stack 和 utility
#include <stack>
#include <utility>

// 定义命名空间 at
namespace at {

// 匿名命名空间，用于实现 SavedTensorDefaultHooksTLS 类的线程局部对象 tls
namespace {
  // 定义线程局部对象 tls，类型为 SavedTensorDefaultHooksTLS
  thread_local impl::SavedTensorDefaultHooksTLS tls;

  // 静态布尔变量 is_initialized，用于标识默认钩子是否已初始化
  // 第一次注册默认钩子时设置为 true，并在整个执行过程中保持为 true
  static bool is_initialized(false);
}

// 断言 SavedTensorHooks 未被禁用的静态函数
static void assertSavedTensorHooksNotDisabled() {
  TORCH_CHECK(SavedTensorDefaultHooks::is_enabled(), tls.disabled_error_message.value());
}

// 返回 SavedTensorHooks 是否启用的方法
bool SavedTensorDefaultHooks::is_enabled() {
  // 查看 NOTE: [disabled_error_message invariant]
  return !tls.disabled_error_message.has_value();
}

// 禁用 SavedTensorHooks 的方法，接受一个禁用消息作为参数
void SavedTensorDefaultHooks::disable(const std::string& message) {
  tls.disabled_error_message = message;
  // 如果栈不为空，则断言 SavedTensorHooks 未被禁用
  if (!tls.stack.empty()) {
    assertSavedTensorHooksNotDisabled();
  }
}

// 启用 SavedTensorHooks 的方法
void SavedTensorDefaultHooks::enable() {
  tls.disabled_error_message = c10::nullopt;
}

// 设置是否跟踪状态的静态方法，返回先前的跟踪状态
/* static */
bool SavedTensorDefaultHooks::set_tracing(bool is_tracing) {
  bool prior  = tls.is_tracing;
  tls.is_tracing = is_tracing;
  return prior;
}

// 获取禁用消息的方法
const std::optional<std::string>& SavedTensorDefaultHooks::get_disabled_error_message() {
  return tls.disabled_error_message;
}

// 获取线程局部状态的方法
const impl::SavedTensorDefaultHooksTLS& SavedTensorDefaultHooks::get_tls_state() {
  return tls;
}

// 设置线程局部状态的方法，接受一个 SavedTensorDefaultHooksTLS 对象作为参数
void SavedTensorDefaultHooks::set_tls_state(const impl::SavedTensorDefaultHooksTLS& state) {
  tls = state;
}

// 懒初始化 SavedTensorHooks 的方法
void SavedTensorDefaultHooks::lazy_initialize() {
  is_initialized = true;
}

// 压栈钩子对象的方法，接受两个 PyObject 指针作为参数
void SavedTensorDefaultHooks::push_hooks(PyObject* pack_hook, PyObject* unpack_hook) {
  // 断言 SavedTensorHooks 已初始化
  TORCH_INTERNAL_ASSERT(is_initialized);
  // 断言 pack_hook 和 unpack_hook 非空
  TORCH_INTERNAL_ASSERT(pack_hook != nullptr && unpack_hook != nullptr);
  // 断言 SavedTensorHooks 未被禁用
  assertSavedTensorHooksNotDisabled();
  // 压栈 pack_hook 和 unpack_hook 对象
  tls.stack.emplace(pack_hook, unpack_hook);
}

// 弹出栈顶钩子对象的方法，返回一个 PyObject* 对象对
std::pair<PyObject*, PyObject*> SavedTensorDefaultHooks::pop_hooks() {
  // 断言 SavedTensorHooks 已初始化且栈非空
  TORCH_INTERNAL_ASSERT(is_initialized && !tls.stack.empty());
  // 弹出栈顶的钩子对象对
  std::pair<PyObject*, PyObject*> hooks = tls.stack.top();
  tls.stack.pop();
  return hooks;
}

// 获取栈顶钩子对象的方法，返回一个 PyObject* 对象对
std::pair<PyObject*, PyObject*> SavedTensorDefaultHooks::get_hooks() {
  // 对于 tls.is_tracing，参见 NOTE: [Deferring tensor pack/unpack hooks until runtime]
  // 如果未初始化或栈为空或正在跟踪，则返回空钩子对象对
  if (!is_initialized || tls.stack.empty() || tls.is_tracing) {
    return std::make_pair(nullptr, nullptr);
  }
  // 返回栈顶的钩子对象对
  return tls.stack.top();
}

} // namespace at
```