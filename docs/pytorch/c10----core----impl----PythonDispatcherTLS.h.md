# `.\pytorch\c10\core\impl\PythonDispatcherTLS.h`

```
#pragma once
// 使用#pragma once确保头文件只被编译一次

#include <c10/core/impl/PyInterpreter.h>
// 包含PyInterpreter.h头文件，用于Python解释器的相关实现

#include <c10/macros/Export.h>
// 包含Export.h头文件，可能用于导出符号或宏定义

namespace c10::impl {

struct C10_API PythonDispatcherTLS {
  // 定义PythonDispatcherTLS结构体，可能用于管理Python调度器的线程局部存储

  static void set_state(PyInterpreter* state);
  // 声明静态成员函数set_state，用于设置Python解释器的状态

  static PyInterpreter* get_state();
  // 声明静态成员函数get_state，用于获取当前Python解释器的状态

  static void reset_state();
  // 声明静态成员函数reset_state，可能用于重置Python解释器的状态
};

struct C10_API DisablePythonDispatcher {
  // 定义DisablePythonDispatcher结构体，可能用于禁用Python调度器

  DisablePythonDispatcher() : old_(PythonDispatcherTLS::get_state()) {
    // 在构造函数中获取当前Python解释器的状态，并将其保存到old_成员变量中
    PythonDispatcherTLS::set_state({});
    // 将Python解释器的状态设置为空，禁用Python调度器
  }

  ~DisablePythonDispatcher() {
    // 析构函数，用于恢复Python调度器的状态
    PythonDispatcherTLS::set_state(old_);
    // 恢复之前保存的Python解释器状态，以实现恢复Python调度器的功能
  }

  PyInterpreter* old_;
  // 成员变量，用于保存构造函数中获取的旧的Python解释器状态
};

} // namespace c10::impl
// 命名空间c10::impl结束
```