# `.\pytorch\aten\src\ATen\PythonTorchFunctionTLS.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/SafePyObject.h>
// 引入 c10 库中的 SafePyObject 头文件
#include <c10/macros/Macros.h>
// 引入 c10 库中的宏定义头文件

namespace at::impl {
// 进入 at::impl 命名空间

enum TorchFunctionDisabledState { ENABLED, SUBCLASSES_DISABLED, ALL_DISABLED };
// 定义枚举类型 TorchFunctionDisabledState，表示 Torch 函数禁用状态的三种可能性：启用、子类禁用、全部禁用

struct TORCH_API PythonTorchFunctionTLS {
  // 定义 PythonTorchFunctionTLS 结构体

  static void set_disabled_state(TorchFunctionDisabledState disabled_state_);
  // 设置当前 Torch 函数的禁用状态

  static TorchFunctionDisabledState get_disabled_state();
  // 获取当前 Torch 函数的禁用状态

  static void push_onto_stack(std::shared_ptr<SafePyObject> mode);
  // 将给定模式推入模式栈中，使用 SafePyObject 的共享指针

  static const std::shared_ptr<SafePyObject> pop_stack();
  // 从模式栈中弹出顶部模式，并返回其共享指针

  static const std::shared_ptr<SafePyObject>& get_stack_at(int64_t idx);
  // 获取模式栈中指定索引处的模式的共享指针

  static int64_t stack_len();
  // 获取模式栈的长度，即模式栈中元素的数量

  static const PythonTorchFunctionTLS& get_state();
  // 获取当前线程局部状态对象的引用

  static void set_state(const PythonTorchFunctionTLS& state);
  // 设置当前线程局部状态对象的值

 private:
  // 私有成员变量

  // The mode TLS is split into
  //   - disabled_state, which says which part of torch function are disabled
  //   - stack_, which is a vector of modes representing the stack of user
  //   defined modes
  // 线程局部存储（TLS）分为两部分：
  //   - disabled_state，用于指示哪些 Torch 函数部分被禁用
  //   - stack_，是一个存储用户定义模式的模式向量
  TorchFunctionDisabledState disabled_state_ =
      TorchFunctionDisabledState::ENABLED;
  // 默认情况下，禁用状态为启用
  std::vector<std::shared_ptr<c10::SafePyObject>> stack_;
  // 模式栈，存储 SafePyObject 的共享指针
};

TORCH_API bool torch_function_mode_enabled();
// 返回当前 Torch 函数模式是否启用的布尔值

} // namespace at::impl
// 结束 at::impl 命名空间
```