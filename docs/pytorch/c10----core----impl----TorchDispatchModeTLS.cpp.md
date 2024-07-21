# `.\pytorch\c10\core\impl\TorchDispatchModeTLS.cpp`

```py
namespace c10::impl {

// 进入 c10::impl 命名空间


thread_local TorchDispatchModeTLS torchDispatchModeState;

// 定义线程局部变量 thread_local 的 TorchDispatchModeTLS 类型对象 torchDispatchModeState


bool TorchDispatchModeTLS::any_modes_set(bool skip_infra_modes) {
  if (!torchDispatchModeState.stack_.empty())
    return true;
  if (!skip_infra_modes) {
    // 遍历非基础模式，检查是否有任何非空值
    for (const auto i : c10::irange(
             static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
      if (torchDispatchModeState.infra_modes_[i] != c10::nullopt) {
        return true;
      }
    }
  }
  // 如果都为空，则返回 false
  return false;
}

// 检查是否有任何模式设置的方法，可选择跳过基础模式


void TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
    std::shared_ptr<PyObject_TorchDispatchMode> mode) {
  if (!any_modes_set()) {
    // 如果当前没有任何模式设置，则将 DispatchKey::Python 和 DispatchKey::PythonTLSSnapshot 设置为 true
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
  // 将非基础模式压入栈中
  torchDispatchModeState.stack_.push_back(std::move(mode));
}

// 将非基础模式推入栈顶的方法，如果当前没有任何模式设置，则设置 Python 和 PythonTLSSnapshot 的 DispatchKey 为 true


const std::shared_ptr<PyObject_TorchDispatchMode> TorchDispatchModeTLS::
    pop_stack() {
  std::shared_ptr<PyObject_TorchDispatchMode> out;
  if (!torchDispatchModeState.stack_.empty()) {
    // 弹出栈顶的非基础模式
    out = torchDispatchModeState.stack_.back();
    torchDispatchModeState.stack_.pop_back();
  } else {
    // 否则，逆序遍历基础模式，找到最高优先级的基础模式并弹出
    for (int64_t i =
             static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS) - 1;
         i >= 0;
         --i) {
      if (torchDispatchModeState.infra_modes_[i].has_value()) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        out = std::move(torchDispatchModeState.infra_modes_[i].value());
        torchDispatchModeState.infra_modes_[i] = c10::nullopt;
        break;
      }
    }
  }
  // 检查是否成功弹出模式
  TORCH_CHECK(out, "trying to pop from empty mode stack");
  // 如果当前没有任何模式设置，则将 DispatchKey::Python 和 DispatchKey::PythonTLSSnapshot 设置为 false
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  return out;
}

// 弹出栈顶模式的方法，如果栈非空则直接弹出，否则弹出最高优先级的基础模式


const std::
    tuple<std::shared_ptr<PyObject_TorchDispatchMode>, TorchDispatchModeKey>
    TorchDispatchModeTLS::pop_highest_infra_mode() {
  // 逆序遍历基础模式，找到最高优先级的基础模式并弹出
  for (int64_t i = static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS) - 1;
       i >= 0;
       --i) {
    if (torchDispatchModeState.infra_modes_[i].has_value()) {
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      auto out_mode = torchDispatchModeState.infra_modes_[i].value();
      torchDispatchModeState.infra_modes_[i] = c10::nullopt;
      // 如果当前没有任何模式设置，则将 DispatchKey::Python 和 DispatchKey::PythonTLSSnapshot 设置为 false
      if (!any_modes_set()) {
        c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
        c10::impl::tls_set_dispatch_key_included(
            DispatchKey::PythonTLSSnapshot, false);
      }
      return std::make_tuple(
          std::move(out_mode), static_cast<TorchDispatchModeKey>(i));
    }
  }
  // 如果未找到任何基础模式，抛出异常
  TORCH_CHECK(
      false, "Called pop_highest_infra_mode, but no infra modes were active.")
}

// 弹出最高优先级基础模式的方法，如果找到则弹出并返回，否则抛出异常说明未找到活动的基础模式
const std::shared_ptr<PyObject_TorchDispatchMode>& TorchDispatchModeTLS::
    get_stack_at(int64_t idx) {
  TORCH_CHECK(idx < stack_len(), "Tried to get stack at idx that's too big");
  // Our "logical" stack includes both:
  // - any user modes (the entire torchDispatchModeState.stack_)
  // - any infra modes (members of torchDispatchModeState.infra_modes_ that are
  // not None)

  // idx == 0 means the "bottom" of the stack, which starts with any infra
  // modes (iterating from lowest-priority to highest-priority).
  auto curr_idx = idx;
  // Iterate over the range of possible mode keys
  for (const auto i :
       c10::irange(static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
    // Check if the infra mode at index i is populated
    if (torchDispatchModeState.infra_modes_[i].has_value()) {
      // If current index matches the requested idx, return the infra mode
      if (curr_idx == 0) {
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        return torchDispatchModeState.infra_modes_[i].value();
      }
      curr_idx -= 1;
    }
  }
  // At this point, return the user mode from the stack based on curr_idx
  return torchDispatchModeState.stack_[curr_idx];
}

int64_t TorchDispatchModeTLS::stack_len() {
  auto stack_len = static_cast<int64_t>(torchDispatchModeState.stack_.size());
  int64_t infra_modes_len = 0;
  // Calculate the total length of the stack including infra modes
  for (const auto i :
       c10::irange(static_cast<size_t>(TorchDispatchModeKey::NUM_MODE_KEYS))) {
    // Count the number of populated infra modes
    if (torchDispatchModeState.infra_modes_[i] != c10::nullopt) {
      infra_modes_len += 1;
    }
  }
  // Return the combined length of stack and infra modes
  return stack_len + infra_modes_len;
}

const std::optional<std::shared_ptr<PyObject_TorchDispatchMode>>
TorchDispatchModeTLS::get_mode(TorchDispatchModeKey mode_key) {
  // Return the infra mode corresponding to mode_key
  return torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)];
}

void TorchDispatchModeTLS::set_mode(
    const std::shared_ptr<PyObject_TorchDispatchMode>& mode,
    TorchDispatchModeKey mode_key) {
  // Ensure that the infra mode for mode_key is not already set
  TORCH_CHECK(
      torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] ==
          c10::nullopt,
      "trying to set the current ",
      to_string(mode_key),
      ", but one already exists");

  // If no modes are currently set, enable dispatch keys for Python and PythonTLS
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }

  // Set the infra mode for mode_key to the provided mode
  torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] = mode;
}

const std::optional<std::shared_ptr<PyObject_TorchDispatchMode>>
TorchDispatchModeTLS::unset_mode(TorchDispatchModeKey mode_key) {
  // Retrieve and unset the current infra mode for mode_key
  auto out = torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)];
  torchDispatchModeState.infra_modes_[static_cast<size_t>(mode_key)] =
      c10::nullopt;
  
  // If an infra mode was unset and no modes are left, disable dispatch keys for Python and PythonTLS
  if (out.has_value() && !any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  }
  // Return the previously set infra mode
  return out;
}

const TorchDispatchModeTLS& TorchDispatchModeTLS::get_state() {
  // Return a reference to the current state of TorchDispatchModeTLS
  return torchDispatchModeState;
}
// 设置 TorchDispatchModeTLS 的状态
void TorchDispatchModeTLS::set_state(TorchDispatchModeTLS state) {
  // 使用 std::move 将状态 state 移动到 torchDispatchModeState 中
  torchDispatchModeState = std::move(state);

  // 如果没有任何模式被设置，则将 DispatchKey::Python 和 DispatchKey::PythonTLSSnapshot 设为 false
  if (!any_modes_set()) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, false);
  } else {
    // 否则，将 DispatchKey::Python 和 DispatchKey::PythonTLSSnapshot 设为 true
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonTLSSnapshot, true);
  }
}

// 检查分发模式是否已启用
bool dispatch_mode_enabled() {
  // 如果 DispatchKey::Python 未被排除并且至少有一个 TorchDispatchModeTLS 模式被设置，则返回 true
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::Python) &&
      TorchDispatchModeTLS::any_modes_set();
}

// 将 TorchDispatchModeKey 转换为字符串表示
std::string to_string(TorchDispatchModeKey mode_key) {
  // 根据 mode_key 的不同值返回相应的字符串表示
  switch (mode_key) {
    case TorchDispatchModeKey::PROXY:
      return "ProxyTorchDispatchMode";
    case TorchDispatchModeKey::FAKE:
      return "FakeTensorMode";
    default:
      return "UNKNOWN_MODE";
  }
}

} // namespace c10::impl


这些注释解释了每行代码的作用和功能，确保了代码的每个部分都得到了清晰的说明。
```