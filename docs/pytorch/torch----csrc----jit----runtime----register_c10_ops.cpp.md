# `.\pytorch\torch\csrc\jit\runtime\register_c10_ops.cpp`

```
#include <ATen/core/ATenOpList.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/record_function.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch::jit {

namespace {

// 从C10的OperatorHandle创建Operator对象
Operator createOperatorFromC10(const c10::OperatorHandle& op) {
  // 返回一个Operator对象，其调用函数使用给定的OperatorHandle调用栈
  return Operator(op, [op](Stack& stack) { op.callBoxed(stack); });
}

// 注册监听器，监视操作符注册和取消注册
class RegistrationListener final : public c10::OpRegistrationListener {
 public:
  // 当操作符注册时调用
  void onOperatorRegistered(const c10::OperatorHandle& op) override {
    if (op.schema().name() == "aten::backward") {
      // aten::backward在register_prim_ops_fulljit.cpp中有手动包装。
      // 我们不应该额外从native_functions.yaml导出c10 aten::backward操作到JIT。
      // 这种特殊处理是因为aten::backward需要AliasAnalysisKind::CONSERVATIVE，
      // 而native_functions.yaml中的所有操作都使用AliasAnalysisKind::FROM_SCHEMA。
      // TODO 找到更好的处理方式。
      return; // 如果是aten::backward操作，则直接返回
    }
    torch::jit::registerOperator(createOperatorFromC10(op));
  }

  // 当操作符取消注册时调用
  void onOperatorDeregistered(const c10::OperatorHandle& op) override {
    if (op.schema().name() == "aten::backward") {
      // 参见onOperatorRegistered中的注释，解释为何排除aten::backward操作
      return; // 如果是aten::backward操作，则直接返回
    }
    torch::jit::deregisterOperator(op.schema());
  }
};

// 注册器结构，负责监听操作符的注册和取消注册
struct Registerer final {
  // 立即调用监听器对所有现有操作符进行注册，并在将来注册新操作符时调用它
  Registerer()
      : listenerRAII(c10::Dispatcher::singleton().addRegistrationListener(
            std::make_unique<RegistrationListener>())) {}
  c10::RegistrationHandleRAII listenerRAII;
};

// 返回Registerer的静态实例，确保其在启动时构造
Registerer& registerer() {
  static Registerer registerer;
  return registerer;
}

// 全局实例，在启动时运行其构造函数
C10_UNUSED Registerer& dummy = registerer();

} // namespace

// 确保c10的registerer被定义
void ensure_c10_registerer_defined() {
  registerer();
}

} // namespace torch::jit
```