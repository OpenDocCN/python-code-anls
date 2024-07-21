# `.\pytorch\torch\csrc\autograd\record_function_ops.cpp`

```
// 包含 ATen 库中的头文件，用于线程本地状态、自定义类型扩展和记录函数
#include <ATen/ThreadLocalState.h>
#include <ATen/cpp_custom_type_hack.h>
#include <ATen/record_function.h>
#include <torch/csrc/autograd/record_function_ops.h>

// 包含 Torch 库中的头文件，用于 JIT 运行时操作符和库注册
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/library.h>

// 定义命名空间 caffe2，用于特定类型的注册
namespace caffe2 {
// 为了使 cpp_custom_type_hack 正常工作，需要注册 at::RecordFunction 类型
// NOLINTNEXTLINE(bugprone-exception-escape)
CAFFE_KNOWN_TYPE(at::RecordFunction);
} // namespace caffe2

// 定义命名空间 torch::autograd::profiler，用于性能分析器相关功能
namespace torch {
namespace autograd {
namespace profiler {

// 创建使用 RecordFunction 的新性能分析作用域，并调用其开始回调
static void record_function_enter(
    const std::string& name,
    const std::optional<std::string>& args,
    at::RecordFunction& rec) {
  // 检查 RecordFunction 是否处于活动状态
  if (rec.isActive()) {
    // 如果需要输入参数并且参数可用，则在调用前记录参数
    if (rec.needsInputs() && args.has_value()) {
      rec.before(
          name, c10::ArrayRef<const c10::IValue>{c10::IValue{args.value()}});
    } else {
      // 否则，只记录名称
      rec.before(name);
    }
  }
}

// 旧版本的签名，使用 cpp_custom_type_hack
static at::Tensor record_function_enter_legacy(
    const std::string& name,
    const std::optional<std::string>& args) {
  // 创建新的 RecordFunction 对象
  auto rec = std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
  // 调用进入记录函数的方法
  record_function_enter(name, args, *rec);
  // 使用 cpp_custom_type_hack 创建 TensorOptions
  return at::cpp_custom_type_hack::create(std::move(rec), at::TensorOptions());
}

// 新版本的签名，使用 custom_class
c10::intrusive_ptr<PythonRecordFunction> record_function_enter_new(
    const std::string& name,
    const std::optional<std::string>& args) {
  // 创建新的 PythonRecordFunction 对象
  auto rec =
      c10::make_intrusive<PythonRecordFunction>(at::RecordScope::USER_SCOPE);
  // 调用进入记录函数的方法
  record_function_enter(name, args, rec->record);
  return rec;
}

// 从 Tensor 中获取 RecordFunction 对象的引用
static at::RecordFunction& getRecordFunctionFromTensor(
    const at::Tensor& handle) {
  // 使用 cpp_custom_type_hack 从 Tensor 中转换为 RecordFunction 引用
  auto& rec = at::cpp_custom_type_hack::cast<at::RecordFunction>(handle);
  return rec;
}

// 结束由 record_function_enter 创建的性能分析作用域
static void record_function_exit(at::RecordFunction& rec) {
  // 调用 RecordFunction 的结束方法
  rec.end();
}

// 旧版本的签名，使用 cpp_custom_type_hack
static void record_function_exit_legacy(const at::Tensor& handle) {
  // 通过 Tensor 获取 RecordFunction 引用，然后调用结束方法
  auto& rec = getRecordFunctionFromTensor(handle);
  record_function_exit(rec);
}

// 新版本的签名，使用 custom_class
static void record_function_exit_new(
    const c10::intrusive_ptr<PythonRecordFunction>& record) {
  // 调用结束记录函数的方法
  record_function_exit(record->record);
}

// 模板函数，用于在 Future 上调用结束回调函数
template <typename Func>
c10::intrusive_ptr<c10::ivalue::Future> _call_end_callbacks_on_fut(
    Func get_record,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut) {
    // 定义一个名为 futureProfilingFunc 的回调函数，用于结束关联的记录函数并返回传入未来值。
    auto futureProfilingFunc =
        [get_record = std::move(get_record)](c10::ivalue::Future& fut) {
          // 获取记录器对象的引用
          auto& rec = get_record();
          // 结束记录
          rec.end();
          // 注意：将此未来对象返回给用户，以确保调用 wait() 后能运行性能分析回调。
          // 为了透明性，必须使此未来对象传播 RPC 未来的值。在此处使用 value() 而不是 constValue()，
          // 以确保传播错误。
          return fut.value();
        };
    // 定义一个未来对象，它在运行性能分析回调后完成。
    auto profiledFut = fut->then(
        at::wrapPropagateTLSState(std::move(futureProfilingFunc)),
        fut->elementType());
    // 返回已经性能分析过的未来对象
    return profiledFut;
// Legacy signature using cpp_custom_type_hack
static c10::intrusive_ptr<c10::ivalue::Future> _call_end_callbacks_on_fut_legacy(
    const at::Tensor& handle,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut) {
  return _call_end_callbacks_on_fut(
      // 返回一个 RecordFunction 的引用，通过 lambda 表达式捕获 handle
      [handle]() -> at::RecordFunction& {
        TORCH_INTERNAL_ASSERT(
            // 断言确保 handle 已定义，否则抛出异常
            handle.defined(),
            "Undefined RecordFunction handle. This can happen if the handle is "
            "not correctly persisted and is destroyed before the future is "
            "realized.");

        // 根据 handle 获取对应的 RecordFunction 对象
        return getRecordFunctionFromTensor(handle);
      },
      fut);
}

// New signature using custom_class
c10::intrusive_ptr<c10::ivalue::Future> _call_end_callbacks_on_fut_new(
    const c10::intrusive_ptr<PythonRecordFunction>& record,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut) {
  return _call_end_callbacks_on_fut(
      // 返回一个 RecordFunction 的引用，通过 lambda 表达式捕获 record
      [record]() -> at::RecordFunction& { return record->record; }, fut);
}

// Internal only, do not use directly, use Python's record_function()
TORCH_LIBRARY_FRAGMENT(profiler, m) {
  m.class_<PythonRecordFunction>("_RecordFunction");

  // 定义旧版本的记录函数进入操作
  m.def(
      "_record_function_enter(str name, str? args=None) -> Tensor",
      &record_function_enter_legacy);
  // 定义新版本的记录函数进入操作，返回 _RecordFunction 对象
  m.def(
      "_record_function_enter_new(str name, str? args=None) -> "
      "__torch__.torch.classes.profiler._RecordFunction",
      &record_function_enter_new);
  // 定义旧版本的记录函数退出操作
  m.def("_record_function_exit", &record_function_exit_legacy);
  // 定义新版本的记录函数退出操作，接受 _RecordFunction 参数
  m.def("_record_function_exit._RecordFunction", &record_function_exit_new);

  // 注册用于 JIT 的操作符，将 profiling 相关操作应用于 future
  torch::jit::registerOperator(torch::jit::Operator(
      "profiler::_call_end_callbacks_on_jit_fut(Tensor x, Future(t) y) -> Future(t)",
      [](jit::Stack& stack) {
        // 弹出输入的 future 和 tensor
        auto fut = jit::pop(stack).toFuture();
        auto tensor = jit::pop(stack).toTensor();
        // 调用旧版本的 _call_end_callbacks_on_fut 函数
        auto profiledFut = _call_end_callbacks_on_fut_legacy(tensor, fut);
        // 返回一个 future，其完成时调用 profiling 回调
        jit::push(stack, std::move(profiledFut));
      },
      c10::AliasAnalysisKind::FROM_SCHEMA));
  // 注册用于 JIT 的操作符，将 profiling 相关操作应用于 future 和 PythonRecordFunction
  torch::jit::registerOperator(torch::jit::Operator(
      "profiler::_call_end_callbacks_on_jit_fut._RecordFunction("
      "__torch__.torch.classes.profiler._RecordFunction x, Future(t) y) -> Future(t)",
      [](c10::Stack& stack) {
        // 弹出输入的 future 和 PythonRecordFunction 对象
        auto fut = torch::jit::pop(stack).toFuture();
        auto tensor =
            torch::jit::pop(stack).toCustomClass<PythonRecordFunction>();
        // 调用新版本的 _call_end_callbacks_on_fut 函数
        auto profiledFut = _call_end_callbacks_on_fut_new(tensor, fut);
        // 返回一个 future，其完成时调用 profiling 回调
        torch::jit::push(stack, std::move(profiledFut));
      },
      c10::AliasAnalysisKind::FROM_SCHEMA));
}
```