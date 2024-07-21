# `.\pytorch\torch\csrc\jit\mobile\model_tracer\CustomClassTracer.cpp`

```py
// 包含头文件 CustomClassTracer.h 和 mutex
#include <torch/csrc/jit/mobile/model_tracer/CustomClassTracer.h>
#include <mutex>

// 命名空间 torch::jit::mobile
namespace torch {
namespace jit {
namespace mobile {

// CustomClassTracer 类的构造函数
CustomClassTracer::CustomClassTracer() {
  // 定义记录回调函数 recorder_cb
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    // 获取函数名
    std::string name = fn.name();
    // 使用锁保护 loaded_classes，将函数名插入 custom_classes
    getLoadedClasses().withLock(
        [&name](CustomClassTracer::custom_classes_type& custom_classes) {
          custom_classes.insert(name);
        });
    return nullptr;
  };

  // 添加全局回调函数，指定作用域为 CUSTOM_CLASS
  handle_ = at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                      .scopes({at::RecordScope::CUSTOM_CLASS}));
}

// 获取 loaded_classes 的静态方法
c10::Synchronized<CustomClassTracer::custom_classes_type>& CustomClassTracer::
    getLoadedClasses() {
  // 定义静态 loaded_classes 变量
  static c10::Synchronized<custom_classes_type> loaded_classes;
  return loaded_classes;
}

} // namespace mobile
} // namespace jit
} // namespace torch
```