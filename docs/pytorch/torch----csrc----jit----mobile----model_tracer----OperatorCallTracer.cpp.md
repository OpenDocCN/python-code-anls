# `.\pytorch\torch\csrc\jit\mobile\model_tracer\OperatorCallTracer.cpp`

```py
#include <torch/csrc/jit/mobile/model_tracer/OperatorCallTracer.h>

namespace torch {
namespace jit {
namespace mobile {

// 定义 OperatorCallTracer 类的构造函数
OperatorCallTracer::OperatorCallTracer() {
  // 使用锁定函数清空被调用操作符的集合
  getCalledOperators().withLock([](std::set<std::string>& called_operators) {
    called_operators.clear();
  });

  // 定义记录器回调函数 recorder_cb
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    // 获取操作符名称
    std::optional<c10::OperatorName> op_name = fn.operator_name();
    if (op_name.has_value()) {
      // 使用锁定函数向被调用操作符的集合中插入操作符名称的字符串形式
      getCalledOperators().withLock(
          [op_name](std::set<std::string>& called_operators) {
            called_operators.insert(c10::toString(*op_name));
          });
    }
    // 返回空指针
    return nullptr;
  };

  // 将记录器回调函数注册为全局回调函数，仅在函数范围内生效
  handle_ = at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                      .scopes({at::RecordScope::FUNCTION}));
}

} // namespace mobile
} // namespace jit
} // namespace torch
```