# `.\pytorch\c10\core\impl\TorchDispatchModeTLS.h`

```py
#pragma once

#include <c10/core/SafePyObject.h>
#include <c10/macros/Export.h>

namespace c10::impl {

// 枚举类型定义，表示 TorchDispatchMode 的不同模式
enum class TorchDispatchModeKey : int8_t {
  FAKE,         // 伪造模式
  PROXY,        // 代理模式
  FUNCTIONAL,   // 功能模式
  NUM_MODE_KEYS // 模式键的数量
};

// 使用 SafePyObject 包装的类型别名，用于表示 TorchDispatchMode
using PyObject_TorchDispatchMode = SafePyObjectT<TorchDispatchModeKey>;

// TorchDispatchMode 的线程本地存储结构体
struct C10_API TorchDispatchModeTLS {
  // 不是不变安全的 API。
  // 它不能接受使用 TorchDispatchModeKey 的基础模式
  // 如果要推送一个基础模式到栈上，我们期望你使用 set_mode
  static void push_non_infra_mode_onto_stack(
      std::shared_ptr<PyObject_TorchDispatchMode> mode);
  
  // 弹出栈顶的模式，
  // 在尝试弹出任何基础模式之前，优先考虑用户模式
  static const std::shared_ptr<PyObject_TorchDispatchMode> pop_stack();
  
  // 返回栈上优先级最高的基础模式，
  // 以及其模式键
  static const std::
      tuple<std::shared_ptr<PyObject_TorchDispatchMode>, TorchDispatchModeKey>
      pop_highest_infra_mode();

  // 获取指定索引处的栈上模式
  static const std::shared_ptr<PyObject_TorchDispatchMode>& get_stack_at(
      int64_t idx);
  
  // 返回栈的长度
  static int64_t stack_len();

  // 获取指定模式键的模式，作为可选项
  static const std::optional<std::shared_ptr<PyObject_TorchDispatchMode>>
  get_mode(TorchDispatchModeKey mode_key);
  
  // 取消设置指定模式键的模式
  static const std::optional<std::shared_ptr<PyObject_TorchDispatchMode>>
  unset_mode(TorchDispatchModeKey mode_key);
  
  // 设置指定模式键的模式
  static void set_mode(
      const std::shared_ptr<PyObject_TorchDispatchMode>& mode,
      TorchDispatchModeKey mode_key);

  // 获取当前状态的 TorchDispatchModeTLS 对象
  static const TorchDispatchModeTLS& get_state();
  
  // 设置 TorchDispatchModeTLS 的状态
  static void set_state(TorchDispatchModeTLS state);

  // 检查是否有任何模式被设置，可选择跳过基础模式
  static bool any_modes_set(bool skip_infra_modes = false);

 private:
  std::vector<std::shared_ptr<PyObject_TorchDispatchMode>> stack_;
  // 用户可以将多个 ProxyTorchDispatchMode 对象推送到栈上
  // 但我们一次只允许将一个 FakeTensorMode 推送到栈上
  // （额外推送 FakeTensorMode 是无操作）
  std::array<
      std::optional<std::shared_ptr<PyObject_TorchDispatchMode>>,
      static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS)>
      infra_modes_;
};

// 检查分发模式是否已启用的全局函数
C10_API bool dispatch_mode_enabled();

// 将 TorchDispatchModeKey 转换为字符串表示的全局函数
C10_API std::string to_string(TorchDispatchModeKey mode_key);

} // namespace c10::impl
```