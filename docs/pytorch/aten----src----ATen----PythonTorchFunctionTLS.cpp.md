# `.\pytorch\aten\src\ATen\PythonTorchFunctionTLS.cpp`

```py
// 包含头文件 PythonTorchFunctionTLS.h 和 TensorImpl.h
#include <ATen/PythonTorchFunctionTLS.h>
#include <c10/core/TensorImpl.h>

// 定义在命名空间 at::impl 内
namespace at::impl {

// 静态线程局部变量 pythonTorchFunctionState，用于保存线程本地状态
static thread_local PythonTorchFunctionTLS pythonTorchFunctionState;

// 将给定的 SafePyObject 模式压入线程本地状态的堆栈中
void PythonTorchFunctionTLS::push_onto_stack(std::shared_ptr<SafePyObject> mode) {
  pythonTorchFunctionState.stack_.push_back(std::move(mode));
}

// 弹出线程本地状态堆栈的顶部元素，并返回其值
const std::shared_ptr<SafePyObject> PythonTorchFunctionTLS::pop_stack() {
  TORCH_CHECK(!pythonTorchFunctionState.stack_.empty(), "trying to pop from empty mode stack");
  auto out = pythonTorchFunctionState.stack_.back();
  pythonTorchFunctionState.stack_.pop_back();
  return out;
}

// 返回线程本地状态堆栈中特定索引处的元素引用
const std::shared_ptr<SafePyObject>& PythonTorchFunctionTLS::get_stack_at(int64_t idx) {
  TORCH_CHECK(idx < static_cast<int64_t>(pythonTorchFunctionState.stack_.size()), "Tried to get stack at idx that's too big");
  return pythonTorchFunctionState.stack_[idx];
}

// 返回线程本地状态堆栈的长度
int64_t PythonTorchFunctionTLS::stack_len() {
  return pythonTorchFunctionState.stack_.size();
}

// 设置线程本地状态的禁用状态
void PythonTorchFunctionTLS::set_disabled_state(TorchFunctionDisabledState disabled_state) {
  pythonTorchFunctionState.disabled_state_ = disabled_state;
}

// 获取线程本地状态的禁用状态
TorchFunctionDisabledState PythonTorchFunctionTLS::get_disabled_state() {
  return pythonTorchFunctionState.disabled_state_;
}

// 设置线程本地状态的状态为给定状态
void PythonTorchFunctionTLS::set_state(const PythonTorchFunctionTLS& state) {
  pythonTorchFunctionState = state;
}

// 获取线程本地状态的当前状态
const PythonTorchFunctionTLS& PythonTorchFunctionTLS::get_state() {
  return pythonTorchFunctionState;
}

// 检查 Torch 函数模式是否已启用
bool torch_function_mode_enabled() {
  return PythonTorchFunctionTLS::get_disabled_state() != TorchFunctionDisabledState::ALL_DISABLED &&
         PythonTorchFunctionTLS::stack_len() > 0;
}

} // namespace at::impl
```