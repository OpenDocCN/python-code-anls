# `.\pytorch\aten\src\ATen\ThreadLocalPythonObjects.cpp`

```py
// 包含头文件 c10/core/TensorImpl.h，提供了有关张量实现的功能
#include <c10/core/TensorImpl.h>

// 包含头文件 ATen/ThreadLocalPythonObjects.h，提供了线程本地 Python 对象的功能
#include <ATen/ThreadLocalPythonObjects.h>

// 包含头文件 c10/util/Exception.h，提供了异常处理的实用功能
#include <c10/util/Exception.h>

// 引入 std::utility 标准库，用于提供通用的编程支持
#include <utility>

// 定义了 at::impl 命名空间，用于封装实现细节
namespace at::impl {

// 定义了静态的线程局部变量 py_objects，用于存储线程本地的 Python 对象
static thread_local ThreadLocalPythonObjects py_objects;

// 设置线程本地 Python 对象的键值对
void ThreadLocalPythonObjects::set(const std::string& key, std::shared_ptr<SafePyObject> value) {
  // 将键值对存入 obj_dict_ 中
  py_objects.obj_dict_[key] = std::move(value);
}

// 获取线程本地 Python 对象的值
const std::shared_ptr<SafePyObject>& ThreadLocalPythonObjects::get(const std::string& key) {
  // 检查是否存在指定键的对象
  TORCH_CHECK(py_objects.obj_dict_.count(key));
  // 返回指定键的对象引用
  return py_objects.obj_dict_[key];
}

// 检查线程本地 Python 对象是否包含指定的键
bool ThreadLocalPythonObjects::contains(const std::string& key) {
  // 检查 obj_dict_ 中是否存在指定键
  return py_objects.obj_dict_.count(key);
}

// 设置线程本地 Python 对象的状态
void ThreadLocalPythonObjects::set_state(ThreadLocalPythonObjects state) {
  // 将传入的状态移动赋值给 py_objects
  py_objects = std::move(state);
}

// 获取线程本地 Python 对象的当前状态
const ThreadLocalPythonObjects& ThreadLocalPythonObjects::get_state() {
  // 返回当前的 py_objects 状态
  return py_objects;
}

} // namespace at::impl
```