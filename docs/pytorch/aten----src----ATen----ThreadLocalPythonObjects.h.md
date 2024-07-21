# `.\pytorch\aten\src\ATen\ThreadLocalPythonObjects.h`

```py
#pragma once
// 指令：确保头文件仅被编译一次

#include <c10/core/SafePyObject.h>
// 包含：引入 SafePyObject 类的头文件
#include <c10/macros/Macros.h>
// 包含：引入宏定义的头文件
#include <unordered_map>
// 包含：引入无序映射容器的头文件

namespace at::impl {

struct TORCH_API ThreadLocalPythonObjects {
  // 结构体：定义线程本地 Python 对象管理器

  static void set(const std::string& key, std::shared_ptr<SafePyObject> value);
  // 方法：设置指定键的 Python 对象

  static const std::shared_ptr<SafePyObject>& get(const std::string& key);
  // 方法：获取指定键的 Python 对象

  static bool contains(const std::string& key);
  // 方法：检查是否包含指定键的 Python 对象

  static const ThreadLocalPythonObjects& get_state();
  // 方法：获取当前线程的状态对象

  static void set_state(ThreadLocalPythonObjects state);
  // 方法：设置线程的状态对象

 private:
  std::unordered_map<std::string, std::shared_ptr<c10::SafePyObject>> obj_dict_;
  // 成员变量：存储键值对映射，键为字符串，值为 SafePyObject 的共享指针
};

} // namespace at::impl
// 命名空间：结束 at::impl 命名空间定义
```