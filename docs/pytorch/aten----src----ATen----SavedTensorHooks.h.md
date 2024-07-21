# `.\pytorch\aten\src\ATen\SavedTensorHooks.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <c10/macros/Export.h>
// 引入 Export.h 头文件，用于导出符号定义

#include <c10/util/Optional.h>
// 引入 Optional.h 头文件，用于提供可选值的支持

#include <c10/util/python_stub.h>
// 引入 python_stub.h 头文件，定义了 PyObject 类型

#include <stack>
// 引入 stack 头文件，用于实现堆栈数据结构

#include <string>
// 引入 string 头文件，用于提供字符串操作支持

#include <utility>
// 引入 utility 头文件，提供一些实用工具

namespace at {

namespace impl {

struct TORCH_API SavedTensorDefaultHooksTLS {
  // 定义结构 SavedTensorDefaultHooksTLS，用于保存张量默认钩子的线程本地状态

  // 堆栈，存储 PyObject* 对象的 pair
  std::stack<std::pair<PyObject*, PyObject*>> stack;

  // 保存禁用错误消息的可选值
  std::optional<std::string> disabled_error_message;

  // 追踪状态标志
  bool is_tracing = false;
};

} // namespace impl

struct TORCH_API SavedTensorDefaultHooks {
  // 定义 SavedTensorDefaultHooks 结构

  // 推入钩子函数的静态方法
  static void push_hooks(PyObject* pack_hook, PyObject* unpack_hook);
  
  // 弹出钩子函数的静态方法
  static std::pair<PyObject*, PyObject*> pop_hooks();
  
  // 获取当前钩子函数的静态方法
  static std::pair<PyObject*, PyObject*> get_hooks();
  
  // 懒初始化函数
  static void lazy_initialize();

  // 获取线程本地状态的静态方法
  static const impl::SavedTensorDefaultHooksTLS& get_tls_state();
  
  // 设置线程本地状态的静态方法
  static void set_tls_state(const impl::SavedTensorDefaultHooksTLS& tls);

  // 禁用 SavedTensorDefaultHooks 的静态方法
  static void disable(const std::string& error_message);
  
  // 启用 SavedTensorDefaultHooks 的静态方法
  static void enable();
  
  // 判断 SavedTensorDefaultHooks 是否启用的静态方法
  static bool is_enabled();
  
  // 获取禁用错误消息的静态方法
  static const std::optional<std::string>& get_disabled_error_message();
};
  
} // namespace at
```